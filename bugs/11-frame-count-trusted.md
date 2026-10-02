# 11. The container's frame count is trusted: an under-reported count drops the last frames

| | |
|---|---|
| Severity | wrong output (missing frames) |
| Status | from code, not reproduced |
| Affected options | video input (`--skip_first_frames`, `--load_cap`, `--chunk_size`, multi-GPU split) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

The number of frames to process comes from OpenCV's `CAP_PROP_FRAME_COUNT`, which for many
containers is an estimate (duration × fps, or a header field). The reader stops after that many
frames even when more can be decoded. If the count is too low the tail of the video is dropped
silently; if it is 0 the file is skipped with a warning. The multi-GPU split uses the same count
for its frame ranges.

## Reproduction

Not run. A file whose header under-reports its length shows it, e.g. a stream cut without
rewriting the header, some VFR or concatenated files, or an MKV/WebM where OpenCV falls back to
duration × fps:

```bash
python - <<'EOF'
import cv2
cap = cv2.VideoCapture("input.mkv")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); real = 0
while cap.read()[0]: real += 1
print("reported", n, "decodable", real)
EOF
python inference_cli.py input.mkv --output out/ --output_format png --model_dir /path/to/models
ls out/input | wc -l    # = reported count, not the decodable one
```

- Expected: every decodable frame (within `--load_cap`) is processed.
- Actual (from the code): exactly `reported − skip` frames; when `reported` is 0, "No frames to
  process" and nothing is written.

## Root cause

`process_single_file` (`inference_cli.py:474-495`):

```python
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
...
frames_to_process = total_frames - args.skip_first_frames
if args.load_cap > 0:
    frames_to_process = min(frames_to_process, args.load_cap)
if frames_to_process <= 0:
    ... "No frames to process ..." ; return 0
```

`_stream_video_chunks` then reads `min(chunk_size, frames_to_process - frames_read)` frames per
chunk and stops at `frames_to_process` (`inference_cli.py:665-670`). `_read_frames_from_cap`
handles an over-reported count gracefully (it stops when `cap.read()` fails,
`inference_cli.py:609-612`), but nothing reads past an under-reported one. Multi-GPU ranges are
computed from the same number (`inference_cli.py:516-521`, `1161-1193`).

## Impact

- Videos with an inaccurate header lose their last frames (or the whole video when the count is
  0), with no error; the output is shorter than the input and audio re-muxing drifts.
- Workaround: remux the input first (`ffmpeg -i in.mkv -c copy fixed.mkv`, or re-encode to an
  intra codec), then check the count with the script above.

## Possible fix

On one GPU, without `--load_cap`, read until the decoder runs out; treat the reported count as a
hint for progress and chunk counts only:

```python
# process_single_file, single-GPU branch (pseudo-code)
frames_to_process = args.load_cap if args.load_cap > 0 else None   # None = until EOF
...
# _stream_video_chunks
while frames_to_process is None or frames_read < frames_to_process:
    read_count = chunk_size if frames_to_process is None else min(chunk_size, frames_to_process - frames_read)
    new_frames = _read_frames_from_cap(cap, read_count)
    if new_frames is None:
        break
```

For multi-GPU, count the frames once (decode-only pass, or `ffprobe -count_frames`) when the
reported count looks unreliable (0, or a container known to estimate). Log a warning whenever the
decoded count differs from the reported one.

Test: a file with a truncated header (or a raw `.mkv` written without cues) must give as many
output frames as `ffprobe -count_frames` reports.

## References

- [cli-flags.md, Input handling](../docs/cli-flags.md#input-handling)
