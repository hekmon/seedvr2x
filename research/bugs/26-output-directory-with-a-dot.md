# 26. `--output` naming a directory with a dot in it is taken for a file: the run fails at the end

| | |
|---|---|
| Severity | crash at save, after all the processing (nothing written) |
| Status | met in a run (2026-10-05, the colour study's ×1.5 clips); path confirmed in the code, both writers checked on CPU |
| Affected options | `--output` with video output (mp4, the default; any `--video_backend`) |
| Version | SeedVR2 `4490bd1` (v2.5.24) |

## Summary

For a video, `process_single_file` treats `--output` as a directory only when the path has no
suffix: `elif not Path(output_path).suffix or (...)` (`inference_cli.py:455`). A directory whose
name contains a dot, such as `out/clip-d1x1.5/`, has one (`Path("out/clip-d1x1.5/").suffix` is
`.5`; the trailing slash doesn't count). The path is then used as the video's file name. The writer
is created only when the first frames are saved, after the model has run. OpenCV's writer can't open a file
with an unknown extension: `ValueError: Cannot create video writer for: out/clip-d1x1.5`
(`inference_cli.py:775`). ffmpeg's ("--video_backend ffmpeg") starts, then exits on "Unable to
choose an output format", which the CLI hides (stderr to `DEVNULL`) behind "ffmpeg process
terminated unexpectedly. Check video path, codec support, and disk space."

## Reproduction

On CPU, the two steps the CLI takes:

```python
from pathlib import Path
import cv2
Path("out/clip-d1x1.5/").suffix                       # '.5': taken for a file name
cv2.VideoWriter("out/clip-d1x1.5", cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48)).isOpened()  # False
cv2.VideoWriter("out/clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 48)).isOpened()     # True
```

`ffmpeg -f lavfi -i testsrc -y out/clip-d1x1.5` fails the same way ("Unable to choose an output
format", exit 234). In a real run: any video input with `--output out/clip-d1x1.5/` processes every
frame, then fails at the save (met by the colour study on its ×1.5 clips).

## Root cause

- The directory test is the absence of a suffix (`inference_cli.py:455`); neither `is_dir()` nor
  a trailing separator is checked, and any suffix counts, not only a video extension.
- The writer is created at the first save (`save_frames_to_video`, `inference_cli.py:767-775`),
  after the processing in the default (non-streaming) path; with `--streaming`, after the first chunk.
- The ffmpeg writer's `isOpened()` polls the process right after starting it, before ffmpeg has
  failed, and its stderr goes to `DEVNULL`.

## Impact

A whole run's work is lost, typically after minutes or hours, for a common naming habit (scale
factors such as `x1.5`, versions such as `v2.1`). PNG output is not affected: for `--output_format
png` the path is always a directory.

## Workaround

Name output directories without a dot, or give a full file name with a video extension
(`--output out/clip-d1x1.5/clip.mp4`).

## Possible fix

Treat the path as a directory when it exists as one, or ends with a separator, or has no known
video extension (`.mp4`, `.mkv`, `.mov`, …); create the writer, or at least validate the output
path, before the processing starts; keep ffmpeg's stderr for the error message.

## How to test

`--output` set to `dir.with.dot/`, an existing directory with a dot, and `dir/file.mp4`: the first
two write `<input>.mp4` inside the directory, the third that file, in a 1-frame run; and a path the
writer cannot open fails before the model loads.
