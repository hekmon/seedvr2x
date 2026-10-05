# FFmpeg: `zscale`'s output depends on the filter thread count

| | |
|---|---|
| Project | FFmpeg, `libavfilter/vf_zscale.c` (not numz) |
| Severity | wrong output: bytes that change with the machine; a 10-bit 4:2:0 read off the exact conversion |
| Status | measured, reproduced twice; **not filed yet** |
| Version | ffmpeg n9.0.2 (static build `n9.0.2-22-g46d8f462ee`, 2026-10-01), its bundled zimg; `vf_zscale.c` read on FFmpeg's master, 2026-10-05 |
| Workaround | `threads=1` on every `zscale` (libavfilter's generic per-filter option) |

## Summary

ffmpeg runs `zscale` in slices: horizontal bands of at least 64 rows, one per filter thread, so by
default one per CPU the process may use (`-filter_threads`), up to 64. Each band gets a zimg graph
of its own. The output then changes with the number of bands, so with the machine:

- **RGB → `yuv420p10le`** (a 4:2:0 master): the chroma near every band edge comes out different, a
  seam at each boundary. About 0.11% of the chroma samples per boundary, up to 13 ten-bit codes;
  luma untouched. Every count from 1 to 16 gives other bytes.
- **`yuv420p10le` → RGB** (reading a 10-bit 4:2:0 source, Catmull-Rom chroma): exact against a
  float64 reference on 1 to 3 bands, off on 99.97% of the samples from 4 bands on (rms 35, up to
  146 sixteen-bit codes, 0.57 of an 8-bit level). Not a seam: the whole picture. An 8-bit 4:2:0
  source reads exactly at any count. Why 10-bit and not 8-bit is not known.

`threads=1` gives the one-band bytes at any `-filter_threads`, both ways.

## Reproduction

ffmpeg and zimg only, one `testsrc2` frame:

```sh
DOWN="zscale=rin=full:pin=709:tin=709:m=709:r=limited:p=709:t=709:d=none:c=left,format=yuv420p10le"
UP="zscale=min=709:rin=limited:cin=left:pin=709:tin=709:m=gbr:r=full:p=709:t=709:d=none:f=bicubic"
UP="$UP:param_a=0:param_b=0.5,format=gbrp16le"
ffmpeg -f lavfi -i testsrc2=s=1920x1080:r=24:d=1 -frames:v 1 -vf format=gbrp16le -c:v ffv1 rgb.mkv

# RGB -> yuv420p10le: 4 different hashes for 1, 2, 4 and 16 bands
for t in 1 2 4 16; do ffmpeg -i rgb.mkv -filter_threads $t -vf "$DOWN" -f framemd5 - | tail -1; done
# with threads=1, 16 filter threads give the 1-band hash
ffmpeg -i rgb.mkv -filter_threads 16 -vf "zscale=threads=1:${DOWN#zscale=}" -f framemd5 - | tail -1

# yuv420p10le -> RGB: 1 and 3 bands agree, 4 and 16 agree, the two pairs differ
ffmpeg -i rgb.mkv -filter_threads 1 -vf "$DOWN" -c:v ffv1 yuv10.mkv
for t in 1 3 4 16; do ffmpeg -i yuv10.mkv -filter_threads $t -vf "$UP" -f framemd5 - | tail -1; done
# threads=1 restores the 1-band hash; an 8-bit yuv420p copy reads the same at any count
```

Measured on a 48-CPU machine: on the frames of a real 1080p clip
([numerics.md](../docs/numerics.md#the-masters-chroma-420-kernels-and-zscales-slices)), and again
with seedvr2x's own conversion chains on a `testsrc2` frame (1, 2, 4 and 16 bands: four masters,
16 = 48; the 10-bit read equal on 1-3 bands, different from 4 on).

| Chain | 2 bands | 4 | 8 | 16 or more |
|---|---|---|---|---|
| RGB → `yuv420p10le`: Cb / Cr samples that differ from 1 band, largest difference (10-bit codes) | 0.12% / 0.11%, 6 | 0.33% / 0.29%, 10 | 0.79% / 0.73%, 10 | 1.70% / 1.58%, 13 |
| `yuv420p10le` → RGB (Catmull-Rom): samples that differ from 1 band (16-bit codes) | 0 | 99.97%, up to 146 | same | same |

## Code path (FFmpeg master)

- `filter_frame`: `s->nb_threads = av_clip(FFMIN(ff_filter_get_nb_threads(ctx), FFMIN(link->h,
  outlink->h) / MIN_TILESIZE), 1, MAX_THREADS)`, with `MIN_TILESIZE` 64 and `MAX_THREADS` 64: a
  1080-row frame gets up to 16 bands.
- `slice_params`: output bands split evenly, their edges aligned to 2 rows; input bands scaled
  from them.
- `graphs_build`: one zimg graph per band. The input band is given as the source's
  `active_region` (the full width, `top` = the band's first row); the output format is the band
  alone (`dst_format.height = out_slice_end - out_slice_start`).
- `filter_slice`: each graph gets the whole input frame and writes its band of the output.

The RGB → 4:2:0 error grows exactly with the number of boundaries, as if the vertical chroma
filter didn't see across a band's edge. The 10-bit 4:2:0 → RGB error covers the whole picture
from 4 bands on, which a seam alone wouldn't; its cause, in ffmpeg's band setup or in zimg's
handling of an active region at 10 bits, isn't established.

## Impact

- The same conversion gives different bytes on machines with different CPU counts, or under a
  different CPU affinity: a lossless master and its checksums aren't reproducible across
  machines, and a job resumed elsewhere writes other bytes, unnoticed.
- A 10-bit 4:2:0 source (10-bit HEVC, FFV1 `yuv420p10le` masters) reads slightly wrong on any
  machine that gives zscale 4 bands or more, which is most of them.
- 8-bit 4:2:0 reads and plain resizes were unaffected in these tests.

## Workaround

`threads=1` on each `zscale` (`zscale=threads=1:...`). It is libavfilter's generic per-filter
option ("Allowed number of threads", beside `enable` and `thread_type`): `ffmpeg -h filter=zscale`
doesn't list it, `ffmpeg -h full` does, and every build takes it. One band costs 2.3–2.6 ms per
1080p frame. seedvr2x forces it on every zscale it runs
([DESIGN.md](../../seedvr2x/DESIGN.md#input)), and [`ffv1_out.py`](../scripts/ffv1_out.py) and
[`fr_clips.py`](../scripts/fr_clips.py) set it.

## Possible fix

Build each band's graph so that its vertical filters read their full support across the band's
edges (extend the source's active region by the filter's reach and crop the output), or fall back
to one band for subsampled formats until that holds. A test: the two chains above, `framemd5` at
1 and at 16 filter threads, must agree.

## Before filing

- Re-run the reproduction on FFmpeg's current git master, and name the zimg version.
- Search FFmpeg's trac for zscale and slice threading.
- Attach the reproduction and the table; mention that `threads=1` restores the one-band output.
