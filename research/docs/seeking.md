# Frame-exact access into long-GOP sources

> Status: **measured** with [`scripts/seek_test.py`](../scripts/seek_test.py) (ffmpeg n9.0.2,
> CPU only, decoders at `-threads 16` on a shared 48-core machine; load averages next to the
> timings), for the open question of the same name in
> [DESIGN.md](../../seedvr2x/DESIGN.md#open-questions). One real DVD (MPEG-2 in MKV, closed GOP)
> was tested; MPEG-2 in VOB or broadcast TS, open-GOP MPEG-2 and open-GOP HEVC come from
> synthetic files only.

In short (10 real sources and 9 synthetic or remuxed files; at each target, 8 consecutive frames
are read and their md5 compared with a decode from the start):

- **Seeking to `n / fps` is never safe:** it missed on 6 of the 10 real sources. One frame off
  where millisecond timestamps stray (3–4 of 40 targets), 1–2 frames early on 36 of 40 where the
  decoder drops 2 frames at the start, up to 142 frames early on a Blu-ray that declares 24/1
  fps while its timestamps run at 24000/1001; and 12–16 of 20 on every MP4, TS and VOB test
  file.
- **Seeking to the exact pts is not enough either.** It comes out late on open-GOP leading
  pictures in MP4 (1–3 frames, 5–6 of 20 targets), in MPEG-TS (up to 47) and MPEG-PS (up to 13;
  10–14 of 20). On two Blu-ray remuxes whose PPS changes in-band, 6.7% and 8.3% of the keyframes
  are not valid entry points: a seek landing there returns **wrong pictures that carry the right
  pts**, or drops frames until the next good keyframe (14 of 28 targets in one passage). Checking
  pts can't see either.
- **Seeking one GOP early, then selecting by pts,** was exact on 538 of 540 targets outside that
  passage, where it still failed 9 of 28; the 2 others are damaged frames that no seek reproduces.
  Seeking 1 s early missed 19 of 540.
- **Decoding from the start and counting** costs 1.2–3.9 min to the end of a 2-hour 1080p H.264
  source (730–2,360 fps), 17 min for a 172 Mbit/s HEVC master (171 fps). It is exact only when
  the reader counts: ffmpeg's `select` restarts its count when ffmpeg rebuilds its filter graph
  mid-stream, and was 14 frames off on the DVD.
- **Recommended:** index every frame in the first pass (pts, a content hash, decoder errors),
  seek one GOP early, select by pts, and check every frame's hash; on a mismatch, start further
  back, and as a last resort decode and count ([Recommended method](#recommended-method)). It
  costs 0.04 s per access on the DVD, 0.14–0.23 s on Blu-ray and web H.264, 1.3 s (median) to
  16 s (worst) on a long-GOP HEVC master. The hash costs 3.0 ms per 1080p frame and thread (md5),
  or 0.47 ms (CRC-32). Decoding is deterministic: two full decodes, and 16 against 1 thread,
  gave the same md5 on every frame.

## Why it matters for seedvr2x

A resumed run restarts at a shot, a window or an output segment, and colour correction reads a
shot's input frames again ([DESIGN.md](../../seedvr2x/DESIGN.md#pause-and-resume)), from
whatever users bring: Blu-ray and web H.264, HEVC, DVD MPEG-2, with B-frames and long or open
GOPs, in MKV, MP4, M2TS or VOB. A frame off by one, or decoded wrongly, breaks the bit-identical
resume and corrects colours against the wrong picture. DESIGN.md's third way, a lossless
intermediate, is not measured here: at the 16-bit FFV1 masters' ≈ 70 MiB per second of 1080p
([output.md](output.md#validation)), a 2-hour source would take about 490 GiB.

## Method

- **Reference:** one full decode per file, `ffmpeg -threads 16 -copyts -i SRC -map 0:V:0
  -fps_mode passthrough -enc_time_base:v demux -f framemd5`: pts (stream time base) and md5 of
  every frame in display order, frame n on line n. Frames = packets everywhere but the M2TS
  excerpt and S8 (one and two leading pictures at the start that the decoder doesn't output).
- **Targets** (40 per source, 20 for S3, S4 and the synthetic files; seeded): first and last 3
  frames, 4 keyframes (2) and their neighbours, first and last leading pictures of as many
  open-GOP keyframes, random.
- **Methods**, each reading 8 frames (`-frames:v 8 -fps_mode passthrough`):

| Method | Input options |
|---|---|
| `a_naive` | `-ss n/fps` |
| `b_pts` | `-ss pts_n − start_time`, the exact pts from the reference. `b_copyts`: the same with `-copyts`. `e_pipe`: the same, read as rawvideo through a pipe, md5 taken in Python |
| `b_abs…` | only when start_time ≠ 0: `-seek_timestamp 1 -ss pts_n`, with and without `-copyts`, and `-ss pts_n` alone (absolute and relative mixed up) |
| `c_1s` | `-copyts -ss pts_n − start_time − 1 s`, then `-vf select='gte(pts,PTS_n)'` |
| `c_gop` | the same from the keyframe before the target's own keyframe (keyframes from the packet scan) |
| `d_count` | decode from the start, `-vf select='gte(n,N)'` (3 targets per file) |

- **Verdict per target:** **exact** (the 8 md5 = reference frames n…n+7), **shifted** by d (=
  n+d…), or **corrupt** (some md5 in no frame of the reference). Times: median of 3 runs, page
  cache warm, the 1-minute load average recorded at each run.
- **Entry points:** `keycheck` stream-copies from exactly one keyframe (`-copyts -ss K
  -copypriorss 0`, NUT through a pipe) into a second ffmpeg and compares every decoded frame with
  the reference; `paramsets` compares every in-band SPS/PPS with the container's extradata
  (`filter_units` + `trace_headers`); `nal` prints the NAL and picture types of a few GOPs.

| | Source | Frames | GOP | B-frames, open GOP | Timestamps |
|---|---|---|---|---|---|
| S1 | Blu-ray remux of a cel-animated film: H.264 High 1080p, 19.6 Mbit/s, MKV | 178,949 | 20 (0.83 s) | 1 B. **Open**: non-IDR I-frames with a recovery-point SEI; 6,381 of 10,925 keyframes have a leading B-frame | ms, ≤ 0.71 ms off the 24000/1001 grid |
| S2 | Blu-ray remux of an animated film: H.264 High 1080p, 16.9 Mbit/s, MKV | 193,636 | IDR every 24 (1.0 s) | ≤ 2 B, closed | ms, ≤ 0.5 ms off |
| S3 | web episode of an animated series: H.264 1080p, 2.0 Mbit/s, MKV | 10,501 | IDR, ≤ 96 (4.0 s) | pyramid of 3 B, closed | ms, ≤ 0.5 ms off |
| S4 | sptenc output segment joining 4 scene segments: HEVC 720p, 31 Mbit/s, MP4 | 1,037 | IDR, up to 987 (41 s) | pyramid of 3 B, closed | 2 steps off the grid, at joins |
| S5 | HEVC upscale master of S1: 1920×1026, 172 Mbit/s, MP4 | 178,949 | IDR, median 78 (3.3 s), max 1,632; last GOP 6,118 | pyramid of 3 B, closed | regular |
| S6 | DVD episode of a live-action sitcom: MPEG-2 720×480, 29.97 interlaced (bottom field first), 16:9 anamorphic, 5.9 Mbit/s, MKV | 40,441 | ≤ 15 (0.5 s); `closed_gop=1`, `broken_link=0` in the 9 GOPs traced | ≤ 2 B, closed | ms, ≤ 0.5 ms off the 30000/1001 grid |
| S7 | Blu-ray episode of an anime series: H.264 High 1080p, 26.3 Mbit/s, MKV | 37,327 | ≤ 24 (1.0 s) | pyramid of 3 B. Open: 798 of 1,881 keyframes have up to 3 leading pictures | ms, ≤ 0.54 ms off |
| S8 | anime OVA episode, 69 min: H.264 High 1080p, 23.7 Mbit/s, MKV | 99,583 | ≤ 26 (1.1 s) | ≤ 2 B (4 declared). Open: 3,223 of 4,763 keyframes have up to 2 leading pictures | ms, ≤ 0.63 ms off |
| S9 | Blu-ray remux of a live-action film declared 24/1 fps: H.264 High 1080p, 17.2 Mbit/s, MKV | 142,477 | IDR every 18 (0.75 s) | ≤ 2 B, closed | run at 24000/1001: see [mechanism 7](#failure-mechanisms) |
| S10 | Blu-ray remux of a live-action film: H.264 High 1080p, 25.6 Mbit/s, MKV | 163,248 | 20 (0.83 s) | 1 B. Open, like S1: 5,193 of 9,851 keyframes have a leading B-frame | ms, ≤ 0.71 ms off |

**Synthetic files** (`synth`), from 2 minutes (2,878 frames) of S3: x264 `open-gop=1` keyint 48
(non-IDR I-frames with a recovery point; 40 of 64 keyframes have up to 3 leading pictures), x265
`open-gop=1` keyint 48 (CRA with up to 4 RASL, 51 of 64), MPEG-2 720×480 GOP 15 (`closed_gop=0`
after the first GOP, 2 leading B-frames); in MKV, MP4 and MPEG-TS (start_time 1.48 s), MPEG-2 in
VOB/MPEG-PS (0.54 s) and TS (1.44 s). Plus 2 minutes of S1 stream-copied into M2TS (1.44 s).

## Results

### Exactness

Targets exact / shifted (frames) / corrupt; `b_copyts`, `e_pipe` and `b_abs` (without `-copyts`)
returned the same frames as `b_pts` on every target of every file.

| File (targets) | `a_naive` | `b_pts` | `c_1s` | `c_gop` | `d_count` |
|---|---|---|---|---|---|
| S1 (40) | 36 / 3 (+1) / 1 | 39 / 0 / 1 | 40 | 40 | 3 / 3 |
| S1, dense-IDR passage (28) | – | 14 / 0 / 14 | 28 | 19 / 0 / 9 | – |
| S2 (40), S3 (20), S5 (40) | all exact | all | all | all | all |
| S4 (20) | 4 / 15 (+1), 1 empty | 20 | 20 | 20 | 3 / 3 |
| S6, DVD (40) | 40 | 40 | 40 | 40 | **0 / 3 (+14)** |
| S7 (40) | 35 / 4 (+1), 1 short | 40 | 40 | 40 | 3 / 3 |
| S8 (40) | 4 / 36 (−2, −1) | 40 | 40 | 40 | 3 / 3 |
| S9 (40) | 4 / 36 (−2 to −142) | 38 / 0 / 2 | 38 / 0 / 2 | 38 / 0 / 2 | 3 / 3 |
| S10 (40) | 35 / 4 (+1, +6), 1 short | 39 / 1 (+6) / 0 | 40 | 40 | 3 / 3 |
| x264 or x265 open GOP, MKV (20 each) | 20 | 20 | 20 | 20 | 3 / 3 |
| x264 open GOP, MP4 (20) | 8 / 12 (+1, +3) | 15 / 5 (+1, +3) | 20 | 20 | 3 / 3 |
| x265 open GOP, MP4 (20) | 8 / 12 (+1, +2) | 14 / 6 (+1, +2) | 20 | 20 | 3 / 3 |
| x264 open GOP, TS (20) | 6 / 14 (+1 to +43) | 8 / 12 (+1 to +43) | 12 / 8 (+1 to +3) | 20 | 3 / 3 |
| x265 open GOP, TS (20) | 4 / 16 (+1 to +47) | 6 / 14 (+1 to +47) | 11 / 9 (+1 to +3) | 20 | 3 / 3 |
| MPEG-2 open GOP, VOB and TS (20 each) | 8 / 12 (+1 to +13) | 8 / 12 (+1 to +13) | 20 | 20 | 3 / 3 |
| S1 excerpt in M2TS (20) | 5 / 15 (+1 to +14) | 10 / 10 (+1 to +14) | 20 | 20 | 3 / 3 |

- **With start_time ≠ 0** (TS, VOB, M2TS), `-seek_timestamp 1 -copyts` was never exact: 17 of 20
  targets late by start_time × fps (+13 frames at 0.54 s, +35 at 1.44 s, +36 at 1.48 s) or more,
  the last 3 empty. An absolute time without `-seek_timestamp`: late by 13 to 82 frames.
- **Corrupt reads were never shifted:** the frames sat at the right positions (with `-copyts`,
  the first carried the target's pts), and 1 to 8 of the 8 pictures were wrong. "Short" reads
  returned one frame fewer than were left before the end.

### Time

Seconds per 8-frame read, ffmpeg's start included: median over targets of each target's median
of 3 runs. A read costs the decode from the keyframe it starts at: S5's worst case is its last
GOP (6,118 frames, 4.3 min); on S4, `c_1s` lands in the previous 987-frame GOP.

| File | `b_pts` | `c_1s` | `c_gop` | `c_gop` worst | Load |
|---|---|---|---|---|---|
| S1 | 0.116 | 0.134 | 0.141 | 0.19 | 21–41 |
| S2 | 0.127 | 0.161 | 0.169 | 0.29 | 14–31 |
| S3 | 0.111 | 0.111 | 0.144 | 0.21 | 4–11 |
| S4 (720p HEVC) | 0.150 | 1.27 | 0.20 | 1.83 | 10–20 |
| S5 (172 Mbit/s HEVC) | 0.663 | 0.794 | 1.313 | 15.7 | 9–34 |
| S6 (DVD) | 0.034 | 0.046 | 0.043 | 0.065 | 15–30 |
| S7 | 0.183 | 0.209 | 0.233 | 0.37 | 14–61 |
| S8 | 0.162 | 0.189 | 0.202 | 0.28 | 18–37 |
| S9 | 0.108 | 0.130 | 0.138 | 0.22 | 12–26 |
| S10 | 0.136 | 0.157 | 0.167 | 0.21 | 18–30 |
| Synthetic, 1080p H.264/HEVC | 0.08–0.13 | 0.09–0.15 | 0.11–0.14 | 0.19 | 3–7 |
| Synthetic, MPEG-2 480p | 0.016–0.017 | 0.017–0.018 | 0.018 | 0.021 | 5–6 |

**Decode-and-count** (2 hours = 172,627 frames at 24000/1001, 215,784 at 30000/1001; a middle
target costs half):

| File | Frames/s (at frame n) | Load | To the end of 2 hours |
|---|---|---|---|
| S1 | 1,002–1,029 (15,703–83,215) | 26 | 2.8–2.9 min |
| S2 | 836–844 (20,321–99,516) | 19 | 3.4 min |
| S3 | 2,250–2,356 (6,222–9,313) | 5 | 1.2–1.3 min |
| S4, 720p | 483–558 (525–923) | 14 | 5.2–6.0 min |
| S5 | 171 (24,607) | 25 | 16.8 min |
| S6, DVD 480i | 2,225–2,406 (1,465–20,806) | 23 | 1.5–1.6 min |
| S7 / S8 / S9 / S10 | 732–779 / 856–920 / 999–1,039 / 768–788 | 37 / 24 / 21 / 23 | 3.7–3.9 / 3.1–3.4 / 2.8–2.9 / 3.7–3.8 min |
| Synthetic MPEG-2, 480p | 15,825–17,079 (2,801) | 5–6 | 10–11 s |

The reference decode, which also hashes every frame, ran at 308–339 fps on every 1080p H.264
file whatever its bitrate (framemd5 hashes in one thread), 181 fps on S5, 1,933 fps on S6.

## Failure mechanisms

1. **`n / fps` is not where the frames are.** MKV stores milliseconds: S1, S7 and S10 stray up to
   0.54–0.71 ms from the exact grid, and `n/fps` missed 3–4 of 40; S2, S3 and S6 stay within
   0.5 ms and never missed. Remuxing carries the rounded values along (the x264 file's MP4 copy
   has pts 0, 672, 1328, 2000… at 1/16000, i.e. 0, 42, 83, 125 ms): there `n/fps` missed 12 of
   20, the exact pts 5. S4 joins four segments at 1/24000 (1001 ticks per frame): the step into
   frame 26 is 991 ticks, which puts every later frame 10 ticks early, and the step into frame
   1013 is 998 (13 early); `n/fps` lands just after those frames: +1 on all 15 targets past the
   first join, no frame at all for the last one. S8's decoder doesn't output the two leading
   pictures of its second keyframe (and logs "illegal short term buffer state detected"), so
   frame n is shown at about (n + 2)/fps: −2 on 33 of 40 targets, −1 on 3. S9: see 7.
2. **MKV: ffmpeg seeks 130 ms early.** `fftools/ffmpeg_demux.c` subtracts 3/23 s from the `-ss`
   point when the demuxer doesn't seek on pts (no `AVFMT_SEEK_TO_PTS`; the Matroska demuxer lacks
   it) and a stream has a B-frame delay. Decoding starts at the keyframe before that point, so
   open-GOP leading pictures (displayed before their keyframe, decoded after it, referencing the
   previous GOP) came out right in every MKV: x264, x265, S1, S7, S8 and S10.
3. **MP4 seeks on decode timestamps.** `libavformat/mov.c` sets `AVFMT_SEEK_TO_PTS` (no 130 ms)
   and searches its sample index by dts after a constant offset ("Here we consider timestamp to
   be PTS, hence try to offset it so that we can search over the DTS timeline",
   `mov_seek_stream`). A leading picture's seek lands on the keyframe after it, the decoder drops
   the picture, and the output starts 1–3 frames late: every `b_pts` miss in MP4 was a frame
   shown just before a keyframe. mov.c's guard (`can_seek_to_key_sample`) only covers HEVC
   samples the file lists as open (a `sync` sample group); our x265 MP4 still missed 6 of 20.
4. **TS and PS have no index.** `mpegts.c` and `mpeg.c` only provide `read_timestamp`: ffmpeg
   bisects on timestamps read from the stream, and decoding started at the next keyframe after
   the position found. The shifts are the rest of the GOP (e.g. +43 = 48 − 5 frames past the
   keyframe); 1 s early still lost leading pictures where GOPs are longer (TS: +1 to +3); one GOP
   early never missed. The MPEG-2 files also have 189 of 2,878 packets without a pts.
5. **start_time, `-copyts` and `-seek_timestamp`.** `-ss` is relative to the file's start_time
   with or without `-copyts` (`ffmpeg_demux.c` adds it unless `-seek_timestamp` is given); TS and
   PS start at 0.54–1.48 s, MKV and MP4 at 0. `-seek_timestamp 1 -ss <absolute>` equals the
   relative form without `-copyts`, but with `-copyts` the accurate-seek trim adds start_time once
   more (`trim_start_us`, same file): the output starts start_time late.
6. **Keyframes that are not entry points: in-band parameter sets.** S1 sends its PPS 0 in-band
   in 53 versions and S10 in 44; the MKV header (extradata) holds one. 732 of S1's 10,925
   keyframes (IDRs too) and 816 of S10's 9,851 carry no SPS/PPS while a PPS unlike the
   extradata's is in effect.
   - A decoder starting there uses the extradata's PPS until the next keyframe carrying SPS/PPS,
     1 to 17 frames later. It outputs wrong pictures ("top block unavailable for requested intra
     mode", "error while decoding MB 0 0"), or none: S10's `b_pts` read that landed on one logged
     "QP 4294967295 out of range" and started 6 frames late. Such errors are a hint, not a
     verdict: reads that started earlier logged 78–120 error lines and came out exact.
   - `keycheck` matched `paramsets` on every keyframe checked: S1, 25 bad of 228 (all 32 between
     305 and 315 s, around a passage with an IDR every one or two frames, and 200 at random); S10,
     9 of 100. Seeking early helps only if the decoder happens to start at a good keyframe: one
     GOP early still failed 9 of the 28 passage targets (1 s early: none).
   - S2, S7, S8 and S9 carry SPS/PPS at every keyframe (S8's SPS changes once, near the end), S3
     and S5 send none in-band: `keycheck` found no bad keyframe there or in S6 (60–100 checked
     each), nor in the synthetic files (30 each), apart from the first-keyframe artifact
     ([Still to test](#still-to-test)).
7. **A declared rate that the timestamps contradict.** S9's Matroska track and ffprobe say 24/1
   fps (default duration 41.667 ms), but its 142,477 frames run from pts 0 to 5,942.438 s (time
   base 1 ms): a mean step of 41.708 ms, i.e. 24000/1001. The steps are 41–42 ms, plus 285
   steps of 62–63 ms, one every ≈ 20.9 s; the frames around the first one show no repeated
   fields (`pic_struct` 0). The audio (DTS 5.1) ends at 5,942.46 s, with the timestamps; at 24/1
   the video would end 5.9 s early. `n/fps` at the declared rate falls up to 142 frames early.
   The 41–42 ms steps average 24 fps; the long ones, from frame 251 and then every 498–502 frames,
   pull the timeline back onto 24000/1001: against that grid anchored at frame 0, every frame lies
   within −10.83 … +10.88 ms, a quarter of a frame, none beyond half. Retimed on decode
   (`fps=24000/1001`, `-fps_mode cfr -r 24000/1001`, or `settb=1001/24000,setpts=N` with
   `-fps_mode passthrough`), the first 3,000 frames come out as the source's own, in order, none
   dropped or doubled (framemd5); ffmpeg n9 refuses `-r` with `-fps_mode passthrough` (measured by
   the design conversation, 2026-10-05). seedvr2x refuses such a file and says how to fix it
   ([DESIGN.md](../../seedvr2x/DESIGN.md#input)).
8. **ffmpeg-side counting resets when ffmpeg rebuilds its filter graph.** S6's first 14 frames
   carry no colour description and the 15th brings BT.601 (`smpte170m`). ffmpeg reconfigures the
   filter graph ("video parameters changed") and `select`'s `n` starts again from 0: `d_count`
   came out 14 frames late on all 3 targets. The framemd5 reference counts at the muxer and is
   unaffected.
9. **Damaged frames depend on where decoding starts.** S9's last frames are damaged (the full
   decode logs "error while decoding MB", "corrupt decoded frame"). Decoding from the start
   conceals them one way, every seek another (one frame differs, with 1 or 16 threads, from the
   last keyframe or a GOP earlier): no seek reproduces them; decode-and-count does.

## Recommended method

1. **Index in the first pass.** The scene-detection pass decodes every frame anyway: record each
   frame's pts (`-copyts`, `-enc_time_base demux`), a hash of the decoded picture, and whether the
   decoder logged errors on it, plus the keyframes' pts from the packet scan (`ffprobe
   -show_entries packet=pts,flags` only demuxes). Count the frames as they arrive: never trust
   `n/fps`, a declared frame rate or ffmpeg-side counting, and keep the source pts for the output
   timing (S9).
2. **Read frames n to m:** `ffmpeg -copyts -ss <pts of the keyframe before n's own keyframe −
   start_time> -i SRC -map 0:V:0 -vf "select='gte(pts,PTS_n)'" -fps_mode passthrough -frames:v
   <m−n+1> -f rawvideo pipe:1`. `-ss` is relative to start_time; no `-seek_timestamp` with
   `-copyts`.
3. **Check every frame's hash** against the index. On a mismatch, read again from one keyframe
   further back, doubling each time; as a last resort, decode from the start and count. A late
   start shows up as a mismatch too: the hash covers shifts and corruption. A frame the first
   pass decoded with errors may only match from the start (mechanism 9).

- **Cost per access:** the `c_gop` times above: 0.04 s on the DVD, 0.14–0.23 s on Blu-ray and
  web H.264, 0.2 s on the 720p HEVC segment, 1.3 s median and 16 s worst on S5. The first try
  would have failed on 2 of 540 targets outside S1's dense-IDR passage (S9's damaged frames), and
  on 9 of 28 inside it.
- **Cost per frame of the first pass,** the hash (`hashcost`, load 16): md5 334 frames/s on one
  thread (3.0 ms per 1080p 8-bit 4:2:0 frame, 1,039 MB/s), 2,640 on 8; CRC-32 (`zlib.crc32`)
  2,140 frames/s on one (0.47 ms, 6,658 MB/s), 12,249 on 8. Against a first pass decoding the
  Blu-ray H.264 at 730–1,040 fps and the web episode at up to 2,356, md5 keeps up with 4 threads
  (8), CRC-32 with 1 (4). Hash what the read returns: the decoder's 8-bit 4:2:0 frames are a
  quarter of the 16-bit RGB the model gets (3.1 against 12.4 MB at 1080p).
- **Alternatives:** decode-and-count, 1.2–17 min per 2-hour source to reach the end (half on
  average), counting in the reader. For H.264/HEVC, a parameter-set scan (`paramsets`, 1–54 s
  per film here, page cache warm) predicts the invalid entry points without hashing, but covers
  only that one mechanism.

## Still to test

- **MPEG-2 beyond one DVD in MKV:** DVD VOB (packets without pts, timestamp jumps at cell and
  VOB boundaries), open-GOP MPEG-2 (the DVD tested is closed), broadcast TS (timestamp wraps,
  missing keyframe flags). The synthetic files only exercised the missing index.
- **Real open-GOP HEVC** (CRA/RASL, BLA): in MKV, in MP4 with and without a `sync` sample group,
  and in TS. The first 4K UHD Blu-ray remux (HEVC Main 10, 3840×2160, HDR10 with Dolby Vision
  profile 7 in the same track, 361,198 frames, MKV) has closed GOPs: an IDR without leading
  pictures every 24 frames, or sooner at cuts, B-runs of up to 4, pts on the 24000/1001 grid
  within 0.6 ms (`info` and `nal`; the packet scan took 3.5 s for 87 GB). It decodes at 204 frames
  per second on 16 threads and 315–326 on 48: a first pass over its 4 h 11 min takes at least 30
  or 19 minutes.
- **The method's own costs:** how often the hash check falls back on real sources, and the hash's
  cost inside seedvr2x's first pass, next to scene detection.
- **Not covered:** deinterlacing and telecine (S6 was read as interlaced frames, as decoded).
- **A `keycheck` quirk:** from a file's first keyframe its "bad" result (S3, synthetic MKV and
  MP4, one each) is an artifact: the stream copy drops the first packets, which have no dts.
- **The tool is ready** for these files: `seek` adds the `-seek_timestamp` variants when
  start_time ≠ 0, `nal` decodes MPEG-2 GOP flags and HEVC NAL types, `targets` includes leading
  pictures, `keycheck` works with any codec (`paramsets`: H.264 and HEVC only).

## Reproduce

```bash
S=scripts/seek_test.py; SRC=/path/to/source.mkv; OUT=out/seek/s1   # one OUT per source
python3 $S info $SRC $OUT            # stream facts, packet scan, GOP analysis
python3 $S nal $SRC $OUT             # NAL / picture types of 3 GOPs (trace_headers)
python3 $S ref $SRC $OUT             # full decode: pts and md5 of every frame
python3 $S targets $OUT --count 40   # 20 for the short files
python3 $S seek $SRC $OUT --repeats 3 --count-at 0.02,0.1,0.5 --count-repeats 3
python3 $S paramsets $SRC $OUT       # H.264/HEVC: keyframes a decoder can't start from
python3 $S keycheck $SRC $OUT --sample 100 --range 305:315   # decode from exactly each keyframe
python3 $S ref $SRC $OUT --name ref2 --threads 1 && python3 $S same $OUT ref ref2
python3 $S synth /path/to/web-episode.mkv out/seek/synth --remux /path/to/bluray-remux.mkv \
  --remux-start 290                  # synthetic open-GOP files + an M2TS excerpt
python3 $S hashcost                  # md5, sha1, blake2b, CRC-32, Adler-32 per 1080p frame
python3 $S report out/seek/*/ out/seek/synth/*/   # Markdown tables, video bitrates
```

S5 used `--count-at 0.02,0.1`; `keycheck --sample 200 --range 305:315` on S1, `--sample 60` on
S3, `--sample 100` on the other real sources, `--sample 30` and `targets --count 20` on each
synthetic file. S1's passage: a hand-written `targets.json` (every other frame from 7,405 to
7,459) in a directory linked to S1's `info.json`, `ref.framemd5`, `packets.csv` and
`paramsets.json`, then `seek --methods b_pts,c_1s,c_gop --repeats 1`. S9's last frames: `seek
--only-kinds last --methods b_pts,d_count --count-at 0.999993 --repeats 1`, and again with
`--methods b_pts,c_gop --threads 1`.
