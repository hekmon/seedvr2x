# Frame-exact access into long-GOP sources

> Status: **measured** with [`scripts/seek_test.py`](../scripts/seek_test.py) (ffmpeg n9.0.2,
> CPU only, decoders at `-threads 16` on a shared 48-core machine; load averages next to the
> timings), for the open question of the same name in
> [DESIGN.md](../../seedvr2x/DESIGN.md#open-questions). One real DVD (MPEG-2 in MKV, closed GOP)
> was tested; MPEG-2 in VOB or broadcast TS, open-GOP MPEG-2 and open-GOP HEVC come from
> synthetic files only.

In short (11 real sources and 9 synthetic or remuxed files; at each target, 8 consecutive frames
are read and their md5 compared with a decode from the start):

- **Seeking to `n / fps` is never safe:** it missed on 6 of the 11 real sources. One frame off
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
- **Seeking one GOP early, then selecting by pts,** was exact on 674 of 676 targets outside that
  passage (all 54 open-GOP leading pictures of a real x265 encode included), where it still
  failed 9 of 28; the 2 others are S9's reads of its last frames, which include its damaged frame
  142475 (mechanism 9). Seeking 1 s early missed 19 of 676.
- **Decoding from the start and counting** costs 1.2–3.9 min to the end of a 2-hour 1080p H.264
  source (730–2,360 fps), 17 min for a 172 Mbit/s HEVC master (171 fps). It is exact only when
  the reader counts: ffmpeg's `select` restarts its count when ffmpeg rebuilds its filter graph
  mid-stream, and was 14 frames off on the DVD.
- **Recommended:** index every frame in the first pass (pts, a content hash, decoder errors),
  seek one GOP early, select by pts, and check every frame's hash; on a mismatch, start further
  back, and as a last resort decode and count ([Recommended method](#recommended-method)). It
  costs 0.04 s per access on the DVD, 0.14–0.23 s on Blu-ray and web H.264, 1.3 s (median) to
  16 s (worst) on a long-GOP HEVC master. The hash costs 3.0 ms per 1080p frame and thread (md5),
  or 0.47 ms (CRC-32). Decoding intact data is deterministic: two full decodes, and 16 against 1
  thread, gave the same md5 on every frame of every source but S9's frame 142475.
- **Damaged frames are not.** A frame the decoder had to conceal can come out differently with the
  decoder's thread count and, in frame- or slice-threaded decodes, from run to run. S9's frame
  142475 gave a different picture at 16, 4, 2 and 1 threads (101466 only at 4, 16 and slice
  threading); in the 11 synthetic damaged copies, 16-thread decodes varied from run to run in 4
  of 8 H.264 and MPEG-2 copies and 2 of 3 HEVC ones. On 1 thread H.264 and MPEG-2 repeated
  exactly in all 8 copies, and in 5 of them 60 of 60 reads started at the damaged frame's
  keyframe or up to 3 before it gave the from-start decode's frames, damaged ones included. HEVC did not repeat on 1
  thread in any of its 3 copies, nor in 780 runs under 13 decoder option sets (2 to 16 runs of
  60 off the commonest decode); it repeated in some copies at other thread counts.
- **Holes:** on the timeline of the declared rate only S8 (the 2 leading pictures of its start) and
  a cut excerpt leave places empty, of 11 real sources, 2 more web episodes, a 4K ProRes clip,
  213 HEVC segments and the synthetic files; S9 at its declared 24/1 has 23,841 one-place holes
  (it runs at 24000/1001: none). **A joined transport stream** keeps its timestamp jump with
  `-copyts` (995.96 s here; 94,439.8 s when the second piece's timestamps are lower) and loses it
  without.

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
- **Holes:** `holes` puts each reference frame at its nearest place on the timeline of the
  declared rate (`r_frame_rate`, anchored at the first frame); a hole is a run of places no frame
  took, and the packets with a pts inside it tell pictures the decoder did not output from frames
  the container does not hold. Frames further than 1 ms from their place sit on no grid of that
  rate.
- **Damaged frames:** `damage` copies a file with bytes overwritten inside chosen packets (8
  random bytes in a P- and a B-frame, or 40 bytes XORed with 0x5A in a P-frame and in the
  packet after it). `redecode` decodes the same stretch (from the start, or from the keyframe
  0 to 3 before a frame) many times per decoder setting, as the first pass runs it (`-copyts
  -debug_ts`), and tells per frame how many different md5 came out, whether the clean
  reference's or a given decode's among them, and which frames the decoder reported ("corrupt
  decoded frame", tied to its frame by the decoder's own `-debug_ts` line). Settings: `-threads`
  16, 4, 2 and 1 (ffmpeg's default thread type, frame threading on top of slice threading) and
  16 with `-thread_type slice`.
- **Joined transport stream:** two 100-frame MPEG-TS pieces (x264 320×240 25 fps, GOP 25, 2
  B-frames, pts from 1.48 s) and the second with its timestamps 1,000 s later (`-output_ts_offset
  1000`), concatenated byte for byte in both orders, then read with and without `-copyts` and the
  options that could bear on it; then 8-frame seeks at 11 targets (the first two frames, frame 50,
  the three before the join, the three after, frames 150 and 192) with four forms of `-ss`.

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
| S11 | the user's x265 encode of a Blu-ray anime episode (x265 4.1, `crf=22 open-gop keyint=240 min-keyint=24 scenecut=40 bframes=4 radl=0`): HEVC Main 1080p, 1.1 Mbit/s, MKV | 34,801 | CRA (one IDR at the start), median 61 (2.5 s), max 240 (10 s) | pyramid of 4 B. **Open**: 435 CRA; only the 14 placed by keyint (240 frames after the previous keyframe) have leading pictures, 54 RASL (12 × 4, 2 × 3); no RADL, BLA or in-band parameter set | ms, ≤ 0.5 ms off |

**Synthetic files** (`synth`), from 2 minutes (2,878 frames) of S3: x264 `open-gop=1` keyint 48
(non-IDR I-frames with a recovery point; 40 of 64 keyframes have up to 3 leading pictures), x265
`open-gop=1` keyint 48 (CRA with up to 4 RASL, 51 of 64), MPEG-2 720×480 GOP 15 (`closed_gop=0`
after the first GOP, 2 leading B-frames); in MKV, MP4 and MPEG-TS (start_time 1.48 s), MPEG-2 in
VOB/MPEG-PS (0.54 s) and TS (1.44 s). Plus 2 minutes of S1 stream-copied into M2TS (1.44 s):
its 2,896 frames decode to S1's frames 6,936 to 9,832, md5 for md5, without 9,831, a B-frame
decoded after 9,832, which the cut left out.

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
| S11, open-GOP HEVC (40) | 40 | 40 | 40 | 40 | 3 / 3 |
| S11, every RASL picture, its CRA and the frames either side (96) | 96 | 96 | 96 | 96 | – |
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
| S11 (1.1 Mbit/s HEVC) | 0.107 | 0.132 | 0.213 | 0.74 | 1.5–4.7 |
| S11, the 96 leading-picture targets | 0.250 | 0.248 | 0.334 | 1.00 | 4–15 |
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
| S11, HEVC | 1,270–1,831 (1,860–34,800) | 4–12 | 1.6–2.3 min |
| Synthetic MPEG-2, 480p | 15,825–17,079 (2,801) | 5–6 | 10–11 s |

The reference decode, which also hashes every frame, ran at 308–339 fps on every 1080p H.264 file
whatever its bitrate (framemd5 hashes in one thread), 181 fps on S5, 1,933 fps on S6, 338 fps on
S11 (HEVC; with 1 decoder thread 300 fps and the same md5 on every frame).

### Holes in the timeline

Frames, places on the timeline of the declared rate (24000/1001, S6 30000/1001), holes, frames
that took a place already taken, and the largest distance of a frame from its place:

| Source | Frames | Places | Holes | Doubled | Largest distance |
|---|---|---|---|---|---|
| S1, S2, S3, S4, S5, S6, S7, S10, S11 | 178,949; 193,636; 10,501; 1,037; 178,949; 40,441; 37,327; 163,248; 34,801 | = frames | 0 | 0 | 0 to 0.71 ms (0.02 frame) |
| S8 | 99,583 | 99,585 | 1 of 2 places, after frame 0 | 0 | 0.63 ms |
| S9 at the declared 24/1 | 142,477 | 142,620 | 23,841 of 1 place | 23,698 | 20.7 ms (0.50 frame): off the grid |
| S9 at 24000/1001 | 142,477 | = frames | 0 | 0 | 10.9 ms (0.26 frame) |
| the 8 synthetic files | 2,878 | = frames | 0 | 0 | 0.5 ms (MPEG-2: 0.017 ms) |
| the M2TS excerpt | 2,896 | 2,897 | 1 of 1 place, after frame 2,894 | 0 | 0.5 ms |
| 2 more web H.264 episodes in MKV | 10,501 each | = frames | 0 | 0 | 0.5 ms |
| a 4K ProRes clip at 24/1, MOV | 6,314 | = frames | 0 | 0 | 0 |
| 213 HEVC 720p segments in MP4 | 37,393 | = frames | 0 | 0 | 4.96 ms (0.12 frame), steps 41.17–41.71 ms |

The 2 web episodes, the ProRes clip and the 213 segments belong to the scene-detection study (they
were decoded there, not in this campaign).

- **S8's hole is in its first two keyframes.** The file stores its first access unit twice, as
  packets at 0 and 125 ms with the same bytes; the two B-pictures between them (42 and 83 ms)
  lead the second copy and refer to a frame the file doesn't have ("Frame num gap", "no
  picture", "illegal short term buffer state detected"). The decoder, with 1 or 16 threads,
  outputs frames at 0, 125, 167 ms…: frame 1 sits 3 places after frame 0, which is the −2 of
  mechanism 1. With `-flags2 +showall` or `-flags +output_corrupt` the decoder outputs pictures
  at 42 and 83 ms (not looked at further); by default it drops them. This is the only hole
  where the container holds packets.
- **The M2TS excerpt's hole is the cut:** its last keyframe's leading B-frame (S1's frame 9,831)
  is left out of the file; no packet lies in the hole.
- **The segments' frames lie up to 4.96 ms from the grid anchored at their first frame,** none
  by more than half a frame: 209 of the 213 within the 1 ms tolerance, 4 beyond it (which the
  tool calls off the grid), all with no hole and no doubled place. Segment timestamps at 1/24000
  are not all on the 1001-tick grid; all 213 declare 24000/1001 (32 have another average rate).
- **Everywhere but S8, S9 at 24/1 and the excerpt, a frame's number is its place** on the
  declared rate's grid, within 0.71 ms (the 4 segments above excepted). S9 is off the grid at
  24/1 and on it at 24000/1001, within a quarter frame (mechanism 7).

### Damaged frames

Cells: different decodes of the same stretch / runs, by the decoder's thread count (default thread
type, frame threading on top of slice threading); "slice" is `-threads 16 -thread_type slice`.
Every synthetic copy's damaged frames, and the frames that refer to them, came out different from
the undamaged file's; S9 has no undamaged decode. The question here is whether they came out the
same way each time. Where a cell is for two files, both gave it.

| File and damage | 16 | slice | 4 | 2 | 1 |
|---|---|---|---|---|---|
| S9, frame 101466: 120 frames from the keyframe 58 frames before it | 4 / 12 | 2 / 6 | 3 / 6 | 1 / 6 | 1 / 6 |
| S9, frame 142475: from the keyframe 91 frames before it to the end of the file (93 frames) | 1 / 12 | 2 / 6 | 1 / 6 | 1 / 6 | 1 / 6 |
| 1080p H.264 MKV, 40 bytes XORed in a P-frame and the next P-frame, 1,200 frames | 2 / 12 | 4 / 6 | 1 / 6 | 1 / 6 | 1 / 6 |
| 720×480 MPEG-2 in TS and in VOB, 40 bytes XORed in a B- and the next P-frame, 1,200 frames | 1 / 12 | 1 / 6 | 1 / 6 | 1 / 6 | 1 / 6 |
| 1080p HEVC MKV, 40 bytes XORed in a P- and the next B-frame, 1,200 frames | 5 / 12 | 1 / 6 | 2 / 6 | 5 / 6 | 5 / 6 |
| 320×240 H.264 MKV, 40 bytes XORed in a P- and the next B-frame, 200 frames | 6 / 24 | 10 / 24 | 7 / 24 | 3 / 24 | 1 / 24 |
| 320×240 MPEG-2 VOB, 40 bytes XORed in one P-frame | 2 / 24 | 2 / 24 | 1 / 24 | 1 / 24 | 1 / 24 |
| 320×240 HEVC MKV, 40 bytes XORed in one B-frame | 3 / 24 | 3 / 24 | 2 / 24 | 5 / 24 | 6 / 24 |
| 1080p H.264 MKV, 8 random bytes in a P- and a B-frame, 2,100 frames | 1 / 12 | 1 / 6 | – | 1 / 6 | 1 / 3 |
| 1080p HEVC MKV, the same | 1 / 12 | 1 / 6 | – | 3 / 6 | 3 / 3 |
| 720×480 MPEG-2 TS, the same | 2 / 12 | 2 / 6 | – | 1 / 6 | 1 / 3 |
| 720×480 MPEG-2 VOB, the same | 1 / 12 | 1 / 6 | – | 1 / 6 | 1 / 3 |

- **S9's damaged frames.** At 101466 the decoder reported the frame in 8 of 12 runs at 16 threads
  and in none of 6 slice-threaded ones; the 16-thread reads gave 9 times the picture of the decode
  from the start and 3 times another one each; 2 and 1 threads gave that picture in 12 of 12
  runs; slice threading gave 2 pictures, neither the from-start one. At 142475 each thread count
  gave a picture of its own, the same in every run (16, 4, 2 and 1 threads: 4 different md5);
  slice threading gave 2.
- **One thread repeats on H.264 and MPEG-2:** 1 decode in 24 of 24 runs (the 320×240 files), 6 of 6
  and 3 of 3 (the larger ones). MPEG-2 also repeated at 2 threads in all 5 of its files, at 4 in
  the 3 where it was tried (the 320×240 file and both 40-byte ones), and at 16 in 3 of 5 (the
  TS and the VOB with 40 XORed bytes, and the VOB with 8 random ones). HEVC repeated on 1 thread
  in none of its 3 files: 6 decodes in 24 runs, 5 in 6, 3 in 3; in some it repeated at other
  counts (the 1080p file with slice threading, 6 of 6; the 8-byte one at 16 threads, 12 of 12).
  With the 40 XORed bytes in a P-frame, all 37 frames from the damage on differed from run to
  run; with one damaged B-frame, only that frame. The decoder logged nothing for any HEVC damage;
  for H.264 it reported the damaged frames with frame threading (both, in every run of 2 files
  and in 10 of 24 runs of the third) and 1 of 2, 1 of 2 and none of 2 with slice threading.
- **Options on 1 thread (HEVC, the 320×240 file, one damaged B-frame, frame 98; 60 runs each):**
  the plain run gave 5 decodes (54 alike); `-err_detect` `crccheck`, `bitstream`, `buffer`,
  `explode`, `ignore_err`, `careful`, `compliant`, `aggressive`, `-flags +output_corrupt`,
  `-flags2 +showall`, `-flags +bitexact` and `-ec 0` gave 3 to 8 each, 2 to 16 runs of 60 away
  from the commonest decode. Only frame 98 varied and no error was logged. This is not a race
  between threads, since it happens on one; the cause was not looked for. An earlier 24 runs of
  the same file and setting gave 6 decodes, 7 alike, so the rate itself varies between series.
- **Seek against start, 1 thread:** five damaged H.264 and MPEG-2 files, decoded whole from the
  start twice, gave the same md5 for every frame; reads of 300 frames starting at the keyframe of
  the damaged frame and at 1, 2 and 3 keyframes before it (3 runs each, 60 reads) gave the
  from-start decode's md5 on every frame of all 60, the damaged ones included. On S9, three
  decodes from the start at 16 threads were identical for all 142,477 frames, and so were two
  at 1 thread; the two settings differ at 142475 alone, where the 1-thread decode has the
  picture of the 1-thread seek and the 16-thread one is not that of the 16-thread seek. HEVC:
  the 1080p file's two from-start decodes differed (36 frames), so no read could match both;
  the 320×240 file's two were identical, and of 4 positions read (3 runs each) 2 matched them in
  3 of 3 runs, 1 in 2 of 3, 1 in none.

### The first pass on one decoder thread

What one decoder thread would cost seedvr2x's first pass (2026-10-21; CPU only, ffmpeg n9.0.2, a
shared 48-core machine at load averages of 2 to 16;
[`scripts/first_pass_threads.py`](../scripts/first_pass_threads.py)). The command is seedvr2x's own
(`seedvr2x.media.scan.command`: every frame's CRC-32, idet, and the shot detector's 48×27 frames on a
pipe, drained here), as it is and with `-threads 1` as an input option, the decoder's; and a decode
alone (`-map 0:v:0 -f null -`) both ways, once each. The passes ran twice each, in the order default,
one thread, one thread, default:

| Source | Frames | Pass, default threads: s (fps) | Pass, one thread: s (fps) | × | Decode alone, default / one thread: s (fps) | × |
|---|---|---|---|---|---|---|
| S9 | 142,477 | 143.4, 142.7 (994, 999) | 1,023.5, 1,035.0 (139, 138) | 7.2 | 134.8 (1,057) / 932.5 (153) | 6.9 |
| S6, DVD | 40,441 | 4.73, 4.66 (8,542, 8,683) | 17.90, 18.22 (2,260, 2,219) | 3.8 | 3.06 (13,237) / 13.05 (3,099) | 4.3 |

- **On one thread the decode is the pass's slow part.** At its default threads ffmpeg spent 1,715 s of
  CPU on S9's pass in 143 s, 12 cores' worth; on one thread less in all, 1,503–1,517 s, at 139 fps, near
  the decoder's own 153. The pass's other work (the hashes, idet, the detector's scale) added 8 s at
  default threads and 97 s on one.
- **The CRC-32s:** S6's four passes gave the same CRC-32 for every frame, and S9's two one-thread
  passes did. Against S9's default-thread passes they differ at exactly 2 frames, the two the decoder
  reports damaged (`error while decoding MB`, `corrupt decoded frame`): 101466, a P-frame with 2,137 of
  its 8,160 macroblocks concealed, gave 4ff72d4e and 0cc1af10 in the two default passes and b2e236d1
  in both one-thread ones; 142475, a B-frame with 3,475 concealed, 6741e597 in both default passes and
  bee5a9f1 in both one-thread ones. At 101466 the two default passes differ from each other, where
  the three 16-thread decodes above were identical: the default threads' picture of it varies from
  run to run.
- **Against a whole first pass, a projection.** seedvr2x's logs of its first passes with TransNetV2 on
  the GPU (2026-10-10 and 10-12, earlier code, the same machine) give S9 158.4 s and 176.6 s and S6
  45.3 s and 38.6 s, the detector's own time 99% of each: the detector set the pace (810–906 fps on S9)
  and the decode chain (994–999 fps here) kept up behind it. On one thread the chain, at 139 fps,
  would set it instead: about 1,029 s for S9 against 158 s, 870 s more; S6's chain (18 s) stays under
  the detector's 45 s. No first pass was run on one thread with the detector.

S9's two damaged frames read the way seedvr2x's reader reads (`-copyts -noaccurate_seek -ss` at a
keyframe's time after the file's start, a `select` on the target's pts), the decoder on one thread,
from the frame's own keyframe (the last whose pts is at most the frame's) and from 1 to 3 keyframes
before it, 3 runs each; the first frame decoded, then the CRC-32 (the start's: b2e236d1 and bee5a9f1):

| Start | 101466 | 142475 |
|---|---|---|
| Its own keyframe | 101462: b2e236d1 ×3 | 142474: e6779f83 ×3 |
| 1 keyframe before | 101444: b2e236d1 ×3 | 142456: bee5a9f1 ×3 |
| 2 before | 101426: b2e236d1 ×3 | 142438: bee5a9f1 ×3 |
| 3 before | 101408: b2e236d1 ×3 | 142420: bee5a9f1 ×3 |

- **101466 gives the start's picture from every start.**
- **142475 doesn't from its own keyframe.** It is the frame right after the file's last keyframe, an
  IDR (`nal_unit_type` 5), decoded after that IDR and the P-frame that follows it, so it can refer to
  nothing before the IDR; yet decoded from the IDR it comes out another way, the same in 3 runs of 3,
  and from 1 to 3 keyframes back as from the start. Its concealed picture depends on where the
  decode started; why, wasn't looked into. A read that starts one keyframe before a frame's own, as
  seedvr2x's does, gets the start's picture there. The 60 reads above, on the synthetic files, matched
  from the damaged frame's own keyframe too: a real file's damage doesn't always.
- **seedvr2x's reads:** reader.py's `-threads 1` stands after `-map … -vf`, an output option: it sets
  its encoders' threads, and the reads' decoder runs at ffmpeg's default. A one-thread decode takes
  `-threads 1` before `-i`, as above.

### A joined transport stream

The pieces: A (pts 1.48–5.44 s) and B (1,001.40–1,005.36 s). `forward` = A then B, `back` = B then
A. The demuxer logs "DTS discontinuity" and "Packet corrupt" at the join in every read, and all
200 frames come out, with the same 200 md5 (framemd5) with and without `-copyts`.

| Read | forward | back |
|---|---|---|
| no option | first 0, last 7.96 s, every step 40 ms | the same |
| `-copyts` | first 1.480 s, last 1,005.360 s; step of **995.96 s** into frame 100 | first 1,001.400 s, last 95,449.158 s; step of **94,439.8 s** into frame 100 |
| `-copyts -start_at_zero` | first 0, last 1,003.88 s, same step | first 0, last 94,447.76 s, same step |
| `-copyts` and `-fflags +igndts`, `+genpts`, `-dts_delta_threshold 1`, `-dts_error_threshold 1`, `-correct_ts_overflow 1` | as `-copyts` | as `-copyts` |
| `-fflags +igndts`, no `-copyts` | the 100 frames of the second piece share one pts (999.92 s), 105 stderr lines | the same, at 94,443.798 s |
| `-dts_delta_threshold 2000`, no `-copyts` | first 0, last 1,003.88 s: the jump stays | as no option |
| the file remuxed to MKV (`-c copy`), then `-copyts` | first 0, last 7.96 s, every step 40 ms | the same |

- **Without `-copyts` ffmpeg removes the jump** ("timestamp discontinuity (stream id=256):
  995920000, new offset= -995920000"): the second piece continues the first at 40 ms.
- **With `-copyts` the jump stays,** and in `back` the second piece's timestamps (1.48 s) fall
  below the first's (1,005.36 s), which the demuxer reads as a 33-bit wrap: they come out 2^33/90000
  = 95,443.7 s later, at 95,445.2 s.
- **Seeks** (8 frames at 11 targets; exact in forward and in back): `-copyts` with `-ss pts −
  start_time` 7 and 5; no `-copyts` with `-ss` the time the reader sees (the continuous one) 8 and
  8; `-seek_timestamp 1 -copyts -ss pts` 0 and 0 (late by start_time × fps, 37 frames, as in
  mechanism 5; in `back` frames 0, 1 and 50 came 100, 99 and 50 frames late, the targets from
  frame 100 on empty); `-seek_timestamp 1 -ss pts` without `-copyts` 6 and 5. The failures of
  that last form: frames 97 to 99 (below), frame 101 of `forward` and frames 101 and 102 of
  `back` one or two frames early, and frame 192 empty in both.
  - **The 3 targets in the GOP before the join** (frames 97 to 99) came out 3, 2 and 1 frames
    late in every form, the start of the next keyframe's GOP, which is mechanism 4.
  - **With `-copyts` the targets after the wrap were not found** in `back`: frames 102, 150 and
    192 came back empty, 100 and 101 were exact. Without `-copyts` all five targets from frame
    100 on were exact. Frame 192 of `forward`, in the file's last GOP, came back empty in three
    of the four forms (all but the reader's time); this was not looked into.
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
   frame n is shown at about (n + 2)/fps: −2 on 33 of 40 targets, −1 on 3 ([the hole in its
   timeline](#holes-in-the-timeline)). S9: see 7.
2. **MKV: ffmpeg seeks 130 ms early.** `fftools/ffmpeg_demux.c` subtracts 3/23 s from the `-ss`
   point when the demuxer doesn't seek on pts (no `AVFMT_SEEK_TO_PTS`; the Matroska demuxer lacks
   it) and a stream has a B-frame delay. Decoding starts at the keyframe before that point, so
   open-GOP leading pictures (displayed before their keyframe, decoded after it, referencing the
   previous GOP) came out right in every MKV: x264, x265, S1, S7, S8, S10 and S11 (all 54 of its
   RASL pictures).
3. **MP4 seeks on decode timestamps.** `libavformat/mov.c` sets `AVFMT_SEEK_TO_PTS` (no 130 ms)
   and searches its sample index by dts after a constant offset ("Here we consider timestamp to
   be PTS, hence try to offset it so that we can search over the DTS timeline",
   `mov_seek_stream`). A leading picture's seek lands on the keyframe after it, the decoder drops
   the picture, and the output starts 1–3 frames late: every `b_pts` miss in MP4 was a frame
   shown just before a keyframe. mov.c's guard (`can_seek_to_key_sample`) only covers HEVC
   samples the file lists as open (a `sync` sample group); our x265 MP4 still missed 6 of 20.
   The decoder's side, on real open-GOP HEVC (S11's `keycheck`, all 436 keyframes): a decode
   started at a CRA outputs none of its RASL pictures (54 of 54), logs nothing at `-v error`,
   and every later frame is exact. The pictures go missing; they never come out wrong.
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
9. **Damaged frames depend on the decoder's thread count, and on HEVC on chance.** S9 has two
   damaged frames (101466 and 142475; the full decode logs "error while decoding MB", "corrupt
   decoded frame"); the concealed picture is the decoder's guess ([Damaged
   frames](#damaged-frames)).
   - **The thread count can change the guess.** At 142475 the decode from the start at 16
     threads (the same in three decodes) gave one picture, a seek at 16 threads another, and 4,
     2 and 1 threads each their own, repeatable in every run; the seek at 1 thread gave the
     from-start 1-thread picture. 101466 came out the same at 1 and 2 threads as in the 16-thread
     decode from the start.
   - **Frame and slice threading add chance on H.264 and MPEG-2.** 101466 came out as 4
     different pictures in 12 reads at 16 threads (the from-start picture 9 times), and the
     decoder reported the frame in 8 of those 12 reads. Of the 8 damaged H.264 and MPEG-2
     synthetic copies, 4 differed from run to run at 16 threads (frame or slice threading), none
     on 1 thread.
   - **On 1 thread H.264 and MPEG-2 repeat, from the start and from the damaged frame's keyframe
     or up to 3 before it:** 60 of 60 such reads gave the from-start decode, damaged frames
     included. On S9 a 1-thread seek and two 1-thread decodes from the start gave the same
     picture at 142475.
   - **HEVC does not repeat on one thread:** its concealed frames changed from run to run in
     every 1-thread series (3 to 8 different decodes in 3 to 60 runs; at other thread counts some
     series repeated). No `-err_detect`, `-flags` or `-ec` setting tried made them repeat (13
     sets, 780 runs), and the decoder logged no error.

   Undamaged data stays deterministic ([In short](#frame-exact-access-into-long-gop-sources)):
   hashes match whatever the thread count and start. A mismatch on a frame the first pass
   decoded with errors may be this and not a wrong seek.
10. **A timestamp jump inside the file: `-copyts` keeps it, the default removes it.** Two MPEG-TS
   pieces joined byte for byte, the second's timestamps 1,000 s above the first's or below them,
   leave a DTS jump that the demuxer logs ("DTS discontinuity", "Packet corrupt") and decodes
   through. Without `-copyts` ffmpeg shifts the second piece to follow the first ("timestamp
   discontinuity … new offset"): 200 frames at 40 ms steps from 0. With `-copyts` the pts keep the
   jump: a step of 995.96 s when the second piece's timestamps are higher, and when they are
   lower, 94,439.8 s, the demuxer taking the fall for a 33-bit wrap and adding 95,443.7 s.
   `-start_at_zero`, `-fflags +genpts`, `-dts_delta_threshold 1`, `-dts_error_threshold 1` and
   `-correct_ts_overflow 1` leave it; `-fflags +igndts` without `-copyts` gives the 100 frames of
   the second piece one pts. `-ss` follows the reader's timeline: without `-copyts` it is the
   continuous one and every target after the join was exact; with `-copyts` the targets after the
   wrap in `back` (from its frame 102) were not found. A first pass that keeps source pts (the
   recommended method) therefore sees the jump in a joined stream; a stream remuxed to MKV (`-c
   copy`) carries none.

## Recommended method

1. **Index in the first pass.** The scene-detection pass decodes every frame anyway: record each
   frame's pts (`-copyts`, `-enc_time_base demux`), a hash of the decoded picture, and whether the
   decoder logged errors on it, plus the keyframes' pts from the packet scan (`ffprobe
   -show_entries packet=pts,flags` only demuxes). Count the frames as they arrive: never trust
   `n/fps`, a declared frame rate or ffmpeg-side counting, and keep the source pts for the output
   timing (S9). With `-copyts` they carry a timestamp jump inside a joined stream (mechanism 10).
2. **Read frames n to m:** `ffmpeg -copyts -ss <pts of the keyframe before n's own keyframe −
   start_time> -i SRC -map 0:V:0 -vf "select='gte(pts,PTS_n)'" -fps_mode passthrough -frames:v
   <m−n+1> -f rawvideo pipe:1`. `-ss` is relative to start_time; no `-seek_timestamp` with
   `-copyts`.
3. **Check every frame's hash** against the index. On a mismatch, read again from one keyframe
   further back, doubling each time; as a last resort, decode from the start and count. A late
   start shows up as a mismatch too: the hash covers shifts and corruption. A frame the first
   pass decoded with errors may match only a read at the first pass's thread count: on 1 thread
   H.264 and MPEG-2 repeat from the damaged frame's keyframe or up to 3 before it, HEVC does not
   (mechanism 9).

- **Cost per access:** the `c_gop` times above: 0.04 s on the DVD, 0.14–0.23 s on Blu-ray and
  web H.264, 0.2 s on the 720p HEVC segment and 0.21 s (0.74 s worst) on the open-GOP x265
  encode, 1.3 s median and 16 s worst on S5. The first try would have failed on 2 of 676 targets
  outside S1's dense-IDR passage (S9's damaged frames), and on 9 of 28 inside it.
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
- **Open-GOP HEVC in MP4** (with and without a `sync` sample group) **and in TS**, and BLA
  pictures: S11 covers real open-GOP HEVC in MKV only. The first 4K UHD Blu-ray remux (HEVC Main
  10, 3840×2160, HDR10 with Dolby Vision profile 7 in the same track, 361,198 frames, MKV) has
  closed GOPs: an IDR without leading pictures every 24 frames, or sooner at cuts, B-runs of up to
  4, pts on the 24000/1001 grid within 0.6 ms (`info` and `nal`; the packet scan took 3.5 s for 87
  GB). It decodes at 204 frames per second on 16 threads and 315–326 on 48: a first pass over its
  4 h 11 min takes at least 30 or 19 minutes.
- **The method's own costs:** how often the hash check falls back on real sources, and the hash's
  cost inside seedvr2x's first pass, next to scene detection.
- **Damaged frames beyond S9 and synthetic damage:** other real files with damage, an HEVC source
  with damage, and why concealed HEVC frames change from run to run on one thread.
- **Joined streams beyond one synthetic pair:** DVD cell and VOB boundaries and broadcast TS with
  real discontinuities and wraps; the test used two 100-frame pieces.
- **Not covered:** deinterlacing and telecine (S6 was read as interlaced frames, as decoded).
- **A `keycheck` quirk:** from a file's first keyframe its "bad" result (S3, S11, synthetic MKV
  and MP4) is an artifact of its stream copy, which keeps every packet but moves every timestamp
  forward by the reorder delay (+83 ms in MKV, whose first 2 packets have no dts; +1328/16000 s
  in MP4, whose first dts is −1328; the NUT output makes the first dts non-negative). Compared in
  order, the frames match the reference (28–29 of 30, the rest is the 30-packet cut).
  `-avoid_negative_ts disabled` on the copy keeps the pts on 5 of these 6 files; on S11 ffprobe
  then can't read the NUT stream back.
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
python3 $S synth /path/to/S3.mkv out/seek/synth --remux /path/to/S1.mkv \
  --remux-start 290                  # synthetic open-GOP files from S3 + an M2TS excerpt of S1
python3 $S hashcost                  # md5, sha1, blake2b, CRC-32, Adler-32 per 1080p frame
python3 $S report $(ls -d out/seek/*/ | grep -v '/synth/$') out/seek/synth/*/   # Markdown tables
python3 $S holes $(ls -d out/seek/*/ | grep -v '/synth/$') out/seek/synth/*/   # empty places
python3 $S damage $SRC $OUT damaged.mkv --packets 1008,1009 --bytes 40 --where 0.3333334 --xor 5a
python3 $S redecode damaged.mkv $OUT --frames 1200 --opts "-threads 16" --runs 12 \
  --opts "-threads 16 -thread_type slice" --runs 6 --opts "-threads 1" --runs 6   # damaged frames
```

S5 used `--count-at 0.02,0.1`; `keycheck --sample 200 --range 305:315` on S1, `--sample 60` on
S3, `--sample 100` on the other real sources, `--sample 30` and `targets --count 20` on each
synthetic file. S1's passage: a hand-written `targets.json` (every other frame from 7,405 to
7,459) in a directory linked to S1's `info.json`, `ref.framemd5`, `packets.csv` and
`paramsets.json`, then `seek --methods b_pts,c_1s,c_gop --repeats 1`. S9's last frames: `seek
--only-kinds last --methods b_pts,d_count --count-at 0.999993 --repeats 1`, and again with
`--methods b_pts,c_gop --threads 1`. S11: `keycheck --sample 436` (every keyframe); the 96
leading-picture targets (every RASL picture, its CRA, the frame after the CRA and the frame before
the RASL run) were written by hand into a `targets.json` in a directory linked to S11's
`info.json`, `packets.csv` and `ref.framemd5`, then `seek --methods
a_naive,b_pts,b_copyts,c_1s,c_gop,e_pipe --repeats 3`; the decode-and-count to the last frame:
`seek --only-kinds last --methods d_count --count-at 0.99999 --count-repeats 3 --repeats 3`.

The damaged-frame series, with `$OUT` the clean file's directory (or one linked to its
`info.json`, `ref.framemd5` and `packets.csv`):

- `damage` without `--packets` damages a P-frame from 35% of the file on and a B-frame from 65% on
  with 8 random bytes (`--seed 6`): the 2,100-frame rows, read with `redecode --frames 2100`, 12
  runs at 16 threads and 6 at each other setting (3 at 1 thread).
- `damage --packets 1008,1009 --bytes 40 --where 0.3333334 --xor 5a` the 1,200-frame rows
  (`redecode --frames 1200`, same runs).
- The 320×240 rows are 200 frames of ffmpeg's `testsrc2` encoded with x264
  (`keyint=48:min-keyint=24:open-gop=1:bframes=3:b-pyramid=normal`), x265 (`bframes=4`, the rest
  alike) and `mpeg2video -b:v 1M -g 15 -bf 2` in VOB, damaged with `--packets 100,101` (H.264) or
  `--packets 100`, `--bytes 40 --where 0.3333334 --xor 5a`, and read with `redecode --runs 24`
  for each of `-threads 16`, `-threads 16 -thread_type slice`, 4, 2 and 1.
- S9's windows: `redecode $SRC $OUT --at 101466 --gops-back 3 --frames 120` and `--at 142440
  --gops-back 3 --frames 200` (the file ends after 93 frames), 12 runs at 16 threads and 6 at
  each other setting.
- The options on 1 thread: `redecode damaged-hevc.mkv $OUT --runs 60` once per `--opts "-threads
  1 -err_detect crccheck"` (and `bitstream`, `buffer`, `explode`, `ignore_err`, `careful`,
  `compliant`, `aggressive`, `-flags +output_corrupt`, `-flags2 +showall`, `-flags +bitexact`,
  `-ec 0`, and none).
- Seek against start: `ffmpeg -threads 1 -copyts -i damaged.mkv -map 0:V:0 -fps_mode passthrough
  -enc_time_base:v demux -f framemd5 start.framemd5` twice, then `redecode damaged.mkv $OUT --ref
  start.framemd5 --at N --gops-back G --frames 300 --opts "-threads 1" --runs 3` for G = 0 to 3.
- S9 from the start on one thread: `ref $SRC $OUT --threads 1 --name ref_t1`, twice, and `cmp` the
  framemd5 files with the 16-thread `ref`.

The first pass on one decoder thread, per source (label `S9` or `S6`), with seedvr2x's Python as
`$P` (the script imports seedvr2x for the command; the measurement ran the same commands through
one-off scripts, and this one gave the same CRC-32 lists on S6's passes and the same CRC-32s on S9's
seeks from the own keyframe and 1 back, the ones run again):

```bash
F=scripts/first_pass_threads.py; O=out/threads
for r in "default 1" "t1 1" "t1 2" "default 2" "decode-default 1" "decode-t1 1"; do
  $P $F run $O S9 $SRC $r; done
$P $F compare $O S9                  # the table, and the frames whose CRC-32 differs
$P $F seek $SRC $O/S9-t1-1.crc $O/S9-seeks.tsv --frames 101466,142475 --back 3 --runs 3
ffmpeg -copyts -ss 5941.55 -i $SRC -map 0:v:0 -c copy -bsf:v trace_headers -f null -   # S9's last IDR
```

The holes of S9 at its real rate: `holes $OUT --rate 24000/1001` (the default is the declared
24/1).

The joined streams: two 100-frame MPEG-TS pieces, `ffmpeg -f lavfi -i
testsrc2=s=320x240:r=25 -frames:v 100 -c:v libx264 -preset veryfast -g 25 -bf 2 -f mpegts a.ts`,
and the same with `,negate` after the source and `-output_ts_offset 1000` into `b.ts`; `cat a.ts
b.ts > forward.ts`, `cat b.ts a.ts > back.ts`. Each file is read with `ffmpeg -hide_banner -v
warning [OPTION] -i FILE -map 0:v:0 -fps_mode passthrough -enc_time_base:v demux -f framecrc -`
for the options of the table, and the pts of the frames compared; for the md5 comparison and the
seeks the same with `-f framemd5`, the seeks being `-ss` in its four forms, 8 frames each, compared
with a full decode. The two pieces need different pictures (hence `negate`), or the md5 of a wrong
place can equal the right one's.
