# Attention in the SeedVR2 DiT: what the backends really do

> Status: **analysis from reading the code** (SeedVR2 `4490bd1`, 7B DiT). The predictions in
> [What to verify](#what-to-verify) still need an instrumented run.

## Windowed attention

Code: `src/models/dit_7b/window.py`, `src/models/dit_7b/nablocks/mmsr_block.py`,
`configs_7b/main.yaml`.

- The latent video (VAE: 8× spatial, 4× temporal) is patchified with `patch_size [1, 2, 2]`. One DiT
  token is therefore 2×2 latent pixels, i.e. **16×16 output pixels**, and one latent frame (≈ 4
  video frames).
- Attention is **3D-windowed**: the token grid `(t, h, w)` is cut into windows, and tokens only
  attend within their window. Each window, plus a copy of the text-prompt tokens
  (`na.repeat_concat_idx`, averaged back afterwards), is one independent sequence. All windows are
  packed into a single **varlen** attention call (`cu_seqlens`).
- Window config: `window: (4,3,3)` for all 36 layers, alternating `720pwin_by_size_bysize`
  (regular) and `720pswin_by_size_bysize` (shifted by half a window, Swin-style) between layers.
- **Window size is defined relative to 720p, so it doesn't change with output resolution.** The
  token grid is rescaled to a 45×80 area, which gives windows of `ceil(45/3) × ceil(80/3)` =
  **15 × 27 tokens** (240 × 432 output pixels) for 16:9 content. In time a window spans
  `wt = ceil(min(t, 30) / 4)` latent frames. **A higher resolution means more windows, not larger
  ones.** Only the batch size, through `t`, changes the window length.

| `--batch_size` | Latent frames `t` | `wt` | Max video tokens per window |
|---|---|---|---|
| 5 | 2 | 1 | 405 |
| 9 | 3 | 1 | 405 |
| 21 | 6 | 2 | 810 |
| 81 | 21 | 6 | 2430 |
| ≥ 117 | ≥ 30 | 8 | 3240 (cap) |

Each sequence also contains the text tokens: a fixed embedding, `pos_emb.pt`, a few dozen tokens.

## Windows are almost never all the same size

The window size rarely divides the token grid, and the shifted layers cut half-windows at the
borders. Example: 1080p output (padded to 1088 → 68 × 120 tokens):

- regular layers: heights 15,15,15,15,**8** × widths 27,27,27,27,**12**, so 4 distinct sizes
- shifted layers: heights **7**,15,15,15,15,**1** × widths **13**,27,27,27,**26**, down to windows
  1 token tall

720p gives widths 27/27/26. 4K (135 × 240 tokens) gives heights 9×15 (even) but widths
8×27 + 24. **Practically every attention call is variable-length.**

## Consequences for `--attention_mode`

- `sageattn_3`: SeedVR2's wrapper (`call_sage_attn_3_varlen`) only uses SA3 when every sequence
  in the call has the same length. Otherwise it falls back to SA2's `sageattn_varlen`.
  **Prediction: SA3 almost never runs**, and `sageattn_3` behaves like `sageattn_2`.
- `sageattn_2`: `sageattn_varlen` is a Triton INT8-QK/FP16-PV kernel. SA2's fast CUDA kernels
  only exist behind the batched API (see [environment.md](environment.md), pitfall 9). Making them
  usable means grouping each call's windows by length and calling the batched API once per group:
  a patch of about 30 lines in `compatibility.py`. The same grouping would let SA3 actually run.
- **Attention is a small share of DiT compute at these lengths.** Per token and per layer, the
  linear and MLP layers cost about `24·d²` FLOPs and attention about `4·L·d`. That makes the
  attention share about `L / (6·d)` with `d = 3072`: **≈ 2% at L ≈ 405** (batch 5–9),
  ≈ 18% at the 3240 cap. So the attention backend should barely move end-to-end DiT time
  at usual batch sizes. That would also explain why micro-benchmarks only showed gains at
  16k-token sequences, a length SeedVR2 never reaches.

## What to verify

Instrumented run (wrap the `call_*_varlen` functions): for each attention call, record the window
length distribution, uniform or not, and which kernel really ran, plus attention's measured share
of DiT time. Do it at 1080p and 4K, with batch sizes 5 and 81. Then decide whether the
length-grouping patch is worth writing.
