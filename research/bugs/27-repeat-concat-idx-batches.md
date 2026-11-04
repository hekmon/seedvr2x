# 27. `repeat_concat_idx` (the windowing's torch.compile rewrite) is ByteDance's at a batch of one only: with several samples it gives the wrong text to each window, and fails on two sizes

| | |
|---|---|
| Severity | wrong output (batch of 2 or more, same size), crash (batch of 2 or more, different sizes); none at a batch of one |
| Status | measured on the RTX 5090 (2026-11-04, SDPA, the sharp 7B in int8 and in fp16); the function's output confirmed on CPU against ByteDance's |
| Affected options | none of the CLI's: it runs a batch of one. Any caller that gives the DiT several samples at once (3B and 7B: `dit_3b/na.py` and `dit_7b/na.py` carry the same function) |
| Version | SeedVR2 `4490bd1` (v2.5.24), `src/models/dit_7b/na.py:320` (and `dit_3b/na.py`); ByteDance's `models/dit_v2/na.py:101-142` at `e4de8c2` |

## Summary

The DiT's windowed attention joins each video window with its sample's text: 7B
`nablocks/mmsr_block.py:123`, 3B `nablocks/attention/mmattn.py:216`,
`na.repeat_concat_idx(vid_len_win, txt_len, window_count)`. numz rewrote the function so that
`torch.compile` can trace it (no `.tolist()`, `_tensor_split` in its place).
The rewrite is equivalent to ByteDance's at a batch of one, where every window has the one text
and `window_count` is its number of windows. With several samples it differs in two ways:

- **Window i gets the text of sample `i % batch`,** where ByteDance gives each sample's own
  windows its own text (`[[x] * n for x, n in zip(txt, txt_repeat)]`). Sample 0's windows
  alternate between the texts of samples 0 and 1.
- **One repeat count for every sample:** `len(vid_len) // batch` (the `else` branch, taken when
  `txt_repeat` holds one value per sample), where ByteDance takes each sample's own. The
  un-concatenation then averages, per sample, text rows that are not that sample's copies, and
  with samples of different sizes (a different number of windows each) the `reshape` in
  `unconcat_coalesce` fails.

The measured effect, on pictures through seedvr2x's DiT (the sharp 7B), seeds 42 to 44: two
720p pictures in one pass, each against itself alone, finish at 31.1 to 35.3 dB PSNR (int8: 31.2
and 34.7; fp16: 31.1 and 35.3). A 720p picture with a 1080p one crashes in
`na.unconcat_coalesce`: `unflatten: sizes [-1, 18] don't multiply up to 1102`.

## Reproduction

The function alone, on CPU, with numz's and ByteDance's side by side (a batch of 2, three
windows each, a text of two rows each; the windows are numbered, the texts are 100-101 and
102-103):

```python
vid_len = torch.tensor([4] * 6); txt_len = torch.tensor([2, 2]); txt_repeat = torch.tensor([3, 3])
vid = torch.arange(24.0)[:, None]; txt = (100 + torch.arange(4.0))[:, None]
concat, _ = repeat_concat_idx(vid_len, txt_len, txt_repeat)
concat(vid, txt)[:, 0].tolist()
```

At a batch of one (`vid_len = [4, 4, 4]`, `txt_len = [2]`, `txt_repeat = [3]`) both give the same
index. With the batch of 2:

- ByteDance's: windows 0-2 followed by 100, 101 and windows 3-5 followed by 102, 103.
- numz's: windows 0, 2 and 4 followed by 100, 101, and windows 1, 3 and 5 followed by 102, 103,
  so sample 0's windows get sample 1's text half the time.

With windows of 3 and 2 (`vid_len = [4] * 5`, `txt_repeat = [3, 2]`), ByteDance's un-concatenation
gives back the two texts; numz's gives five rows (100, 100.5, 101, 102, 103) for four, the
averages taken over the wrong copies.

The picture-level probe (measured by the image server's conductor on 2026-11-04, a script that is
not in this repository): the pictures are frames of Sol Levante at 640x360 and 960x540 (outputs
1280x720 and 1920x1080), the DiT given two samples in one pass, the VAE as usual.

## Root cause

- `repeat_concat_idx` builds the index with `batch_idx = i % batch_size` for window `i` and a
  single `num_repeats = len(vid_len) // batch_size`; ByteDance's builds it with
  `repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat)`, one count a sample, and
  coalesces with the same counts (`repeat_txt_len = (txt_len * txt_repeat).tolist()`).
- With a batch of one, `i % 1 == 0` and the count is the window count: the two agree, which is
  all the CLI and the image server run, so no test comparing outputs at a batch of one can see it.

## Impact

None on seedvr2x's output today: it runs a batch of one. Any use of several samples in one DiT
pass is wrong, whatever the attention backend. The batch is what the DiT time is worth saving
on: two pictures in one pass took 8 to 11% less DiT time than alone (two 720p: 0.766 s against
0.837 s, int8; a 720p and a 1080p: 1.199 s against 1.297 s, with ByteDance's function).

## Workaround

A batch of one. Or ByteDance's function in place of numz's, which makes mixed batches run (it
needs `.tolist()`, so a graph break under `torch.compile`).

## Possible fix

ByteDance's `repeat_concat_idx` (`models/dit_v2/na.py:101-142`), kept as is, or a rewrite for
`torch.compile` that keeps each sample's own text and own count and is held to ByteDance's index
on mixed batches (with samples of equal and of different sizes).

Even with ByteDance's function, a picture in a batch is not its picture alone: 43.3 and 48.4 dB
for two 720p pictures, 48.4 and 44.6 dB for a 720p with a 1080p (int8), a picture paired with
itself differing from alone as well. The batch's shapes change the kernels' reductions; int8's
integer products do not make it hold. Alone, with ByteDance's function patched in, a picture is
numz's bit for bit (a picture's pass alone is bit-stable, run twice: equal).

## How to test

The function on the batches above, numz's against ByteDance's: equal indices and equal
un-concatenation for a batch of one, for equal sizes and for different sizes. Then the DiT: two
pictures in a pass, each compared with its pass alone (expect a small difference, not 31 dB), a
720p with a 1080p running, and one picture's pass bit for bit what it was.
