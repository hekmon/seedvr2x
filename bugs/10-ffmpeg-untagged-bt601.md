# 10. `--video_backend ffmpeg` writes untagged BT.601 YUV: colour shift in players that assume BT.709 for HD

| | |
|---|---|
| Severity | wrong output (playback colours) |
| Status | measured (the encoder command alone, on a synthetic frame) |
| Affected options | `--video_backend ffmpeg` (with or without `--10bit`) |
| Version | SeedVR2 `4490bd1` (v2.5.24), ffmpeg on `PATH` |

## Summary

The ffmpeg writer pipes RGB24 to x264/x265 without a colour matrix or any colour tag. swscale
converts with its default BT.601 matrix, and the stream says nothing. Players that assume BT.709
for untagged HD content then decode with the wrong matrix: saturated greens lose 15% and
mid-tones shift by up to 9 levels. ffmpeg's errors are also discarded.

## Reproduction

The writer's exact command (`inference_cli.py:170-175`), on a 1920×1080 frame of colour
patches, then decoded with each matrix:

```bash
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 24 -i patches.rgb \
  -c:v libx264 -pix_fmt yuv420p -preset medium -crf 12 out.mp4
ffprobe -v error -show_entries stream=color_space,color_primaries,color_transfer,color_range out.mp4
ffmpeg -i out.mp4 -frames:v 1 -vf scale=in_color_matrix=bt709:in_range=tv -f rawvideo -pix_fmt rgb24 dec709.rgb
```

Measured (patch centres, 8-bit RGB):

| Patch | Input | Decoded as BT.601 | Decoded as BT.709 (HD player) | Same, with the fix below, as BT.709 |
|---|---|---|---|---|
| red | 255, 0, 0 | 253, 0, 0 | 255, 23, 0 | 254, 0, 0 |
| green | 0, 255, 0 | 0, 255, 0 | 0, 215, 0 | 0, 254, 0 |
| blue | 0, 0, 255 | 0, 0, 254 | 0, 14, 255 | 0, 0, 254 |
| skin | 200, 150, 120 | 199, 149, 118 | 204, 152, 116 | 199, 148, 119 |
| teal | 40, 140, 160 | 40, 138, 158 | 31, 130, 160 | 38, 137, 159 |
| grey | 128, 128, 128 | 128, 128, 128 | 128, 128, 128 | 128, 128, 128 |

`ffprobe`: `color_range=unknown`, `color_space=unknown`, `color_transfer=unknown`,
`color_primaries=unknown`.

- Expected: a stream encoded and tagged as BT.709 (HD), so every player decodes it as encoded.
- Actual: BT.601 coefficients, no tags; correct only in players that guess BT.601.

## Root cause

`FFMPEGVideoWriter.__init__` (`inference_cli.py:166-175`):

```python
self.proc = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
     '-c:v', codec, '-pix_fmt', pix_fmt, '-preset', 'medium', '-crf', '12', path],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
```

- No `-vf scale=out_color_matrix=…`, no `-colorspace` / `-color_range` / `-color_primaries` /
  `-color_trc`: swscale's RGB → YUV default is BT.601, limited range, and nothing is written in
  the bitstream's VUI.
- Untagged HD is commonly decoded as BT.709 (the HD standard), so the matrix mismatch shows as a
  hue/saturation shift on saturated colours; greys are unaffected.
- `stderr=subprocess.DEVNULL`: an encoder failure (unknown codec, full disk, bad size) only
  leaves "ffmpeg exited with code N" as a WARNING in `release()` (`inference_cli.py:201-208`).
- The input side has the same blind spot: OpenCV decodes the source to 8-bit RGB with its own
  matrix choice and drops any colour metadata ([cli-flags.md](../docs/cli-flags.md#input-handling)).

## Impact

- Every `--video_backend ffmpeg` output viewed in a player or browser that applies BT.709 to
  untagged HD: greens and teals visibly shifted. Whether a given player guesses 601 or 709
  varies, which is the problem with untagged streams.
- Workaround: re-tag without re-encoding is not enough (the coefficients are 601); either convert
  (`ffmpeg -i out.mp4 -vf scale=in_color_matrix=bt601:out_color_matrix=bt709 ...`), or write PNG
  and encode yourself with explicit colour options.

## Possible fix

Convert with an explicit matrix and tag the stream:

```diff
--- a/inference_cli.py
+++ b/inference_cli.py
@@ -170,8 +170,12 @@ class FFMPEGVideoWriter:
         self.proc = subprocess.Popen(
             ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
              '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
+             '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
+             '-colorspace', 'bt709', '-color_primaries', 'bt709',
+             '-color_trc', 'bt709', '-color_range', 'tv',
              '-c:v', codec, '-pix_fmt', pix_fmt, '-preset', 'medium', '-crf', '12', path],
-            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
+            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
         )
```

and print `stderr` when the exit code is non-zero (read it in `release()`; with `stderr=PIPE`,
also drain it in a thread or use a temporary file, so that a chatty ffmpeg can't block on a full
pipe).

Measured with these options: `color_space=bt709`, `color_range=tv`, and the BT.709 decode matches
the input within 2 levels (last column above). `color_primaries` / `color_transfer` still read
`unknown` in that test: they may need `-x264-params colorprim=bt709:transfer=bt709` (and the x265
equivalent); check with `ffprobe`.

Better still, propagate the source's colour properties (`ffprobe` the input) instead of assuming
BT.709, which is wrong for SD sources.

Test: the patch frame above through `FFMPEGVideoWriter`, then `ffprobe` (tags) and a BT.709 decode
within 2 levels of the input; a forced failure (output in a read-only directory) must show
ffmpeg's message.

## References

- [cli-flags.md, Output](../docs/cli-flags.md#output)
