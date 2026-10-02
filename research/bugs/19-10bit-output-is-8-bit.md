# 19. `--10bit` encodes 8-bit frames, and is silently ignored with the OpenCV backend

| | |
|---|---|
| Severity | UX-doc (the option doesn't deliver what it promises) |
| Status | from code |
| Affected options | `--10bit`, `--video_backend` |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

`--10bit` switches the ffmpeg writer to x265 `yuv420p10le`, "reduces banding" says the help. But
the frames given to ffmpeg are already 8-bit RGB (`uint8`, truncated, [09](09-uint8-truncation.md)),
and before that they live in a bfloat16 tensor with 8 significant bits. The 10-bit encode only
avoids the rounding of the RGB → YUV step. With `--video_backend opencv` (the default) the flag
is ignored without a word.

## Reproduction

```bash
python inference_cli.py input.mp4 --output out.mp4 --model_dir /path/to/models \
  --video_backend ffmpeg --10bit
ffprobe -v error -show_entries stream=codec_name,pix_fmt out.mp4      # hevc, yuv420p10le
ffmpeg -i out.mp4 -frames:v 1 -f rawvideo -pix_fmt rgb48le frame.rgb48   # then count distinct values per channel
python inference_cli.py input.mp4 --output out2.mp4 --model_dir /path/to/models --10bit
ffprobe -v error -show_entries stream=codec_name,pix_fmt out2.mp4     # mpeg4, yuv420p: flag ignored
```

- Expected: a 10-bit output that keeps the precision the pipeline has (gradients with more than
  256 levels per channel); an error or warning when the backend can't do it.
- Actual (from the code): 10-bit YUV holding 8-bit RGB values (≤ 256 distinct levels per channel
  before the YUV conversion); with OpenCV, an 8-bit `mp4v` file and no message. Not run.

## Root cause

1. `final_video` is allocated in the compute dtype, bfloat16 (`src/core/generation_phases.py:879`),
   and Phase 4 normalizes in place in that dtype (`generation_phases.py:1348`). bf16 has 8
   significant bits: in [0.5, 1) its step is 1/256, i.e. one 8-bit level.
2. The writers convert to `uint8` first (`inference_cli.py:763`):
   ```python
   frames_np = (frames_tensor.cpu().numpy() * 255.0).astype(np.uint8)
   ```
3. `FFMPEGVideoWriter` feeds `-pix_fmt rgb24` and only changes the output format
   (`inference_cli.py:166-175`):
   ```python
   pix_fmt = 'yuv420p10le' if use_10bit else 'yuv420p'
   codec = 'libx265' if use_10bit else 'libx264'
   ```
4. With `opencv`, `save_frames_to_video` never looks at `use_10bit`
   (`inference_cli.py:769-773`); `main()` only validates that ffmpeg exists when
   `--video_backend ffmpeg` is chosen (`inference_cli.py:1535-1539`).

The VAE also decodes in bf16, so the decoded signal itself has ≈ 8–9 significant bits; a true
10-bit output needs float32 (or float16) from the VAE output on, not only at the writer.

## Impact

- Users who pick `--10bit` against banding (dark gradients, anime skies) get x265's slower
  encode and a larger file for the same 8-bit steps. The help's streaming example uses it.
- With the default backend, users believe they have 10-bit output and don't.
- Workaround: none in the CLI for real 10-bit. `--10bit` with ffmpeg still avoids one 8-bit
  rounding (RGB → YUV), which can help slightly; with OpenCV add `--video_backend ffmpeg`.
  [`ffv1_out.py`](../scripts/ffv1_out.py) skips the uint8 step: it stores the bf16 frames as
  16-bit (or 10-bit) RGB. Below 0.5 the bf16 grid is finer than 8 bits: on dark 1080p frames the
  master kept 263–347 distinct values per channel, the CLI's PNG 134–218
  ([output.md](../docs/output.md#validation)).

## Possible fix

1. Warn (or error) when `--10bit` is set with `--video_backend opencv`, in `main()`.
2. Feed ffmpeg 16-bit RGB when `--10bit` is set:
   ```diff
   -            ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
   +            ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le' if use_10bit else 'rgb24',
   ```
   with `frames = (x.float() * 65535).round().clamp(0, 65535).to(torch.uint16)` (or
   `.numpy().astype('<u2')`) instead of the uint8 conversion, and the BGR → RGB swap done on the
   16-bit array.
3. Keep precision upstream: allocate `final_video` in float16 (10 significant bits, same memory as
   bf16), do Phase 4's colour correction and normalization in float32, and cast the VAE output to
   float32 before the `[-1, 1] → [0, 1]` step. The VAE's own bf16 output remains the limit.
4. Document what `--10bit` gives.

Risks: float16 `final_video` keeps the same RAM as bf16; float32 doubles it (≈ 50 MB per 1080p
frame). rgb48le doubles the pipe bandwidth to ffmpeg.

Test: a synthetic dark gradient (≥ 1024 steps across the frame) through `--color_correction
none --10bit`; count distinct levels in the decoded rgb48 output before and after the fix.

## References

- [cli-flags.md, Output](../docs/cli-flags.md#output), [Input and output](../docs/cli-flags.md#input-and-output)
