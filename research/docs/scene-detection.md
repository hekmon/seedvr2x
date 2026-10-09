# Scene detection

> Status: **rounds 1 and 2 labelled, brief and addendum written** (2026-10-05), for DESIGN.md's
> open question on seedvr2x's own scene detection
> ([DESIGN.md](../../seedvr2x/DESIGN.md#open-questions)): which detector, which threshold, what
> to do with bursts, and whether shots need a minimum length.
> Three detectors ran on every frame of 11 sources; the user labelled 100 of their candidates,
> drawn where the detectors disagree ([round 1](#round-1-targeted-on-the-disagreements),
> [estimates](#labelled-estimates)). The [decision brief](#decision-brief) is joint with question
> 2 ([cuts.md](cuts.md)). Tools:
> [`scripts/scd_scores.py`](../scripts/scd_scores.py) (scores, detectors, candidates,
> statistics, label estimates), [`scripts/scd_review.py`](../scripts/scd_review.py) (review
> sheets and index).

In short (the statistics count detections; the estimates come from the labels):

- **Labelled, TransNetV2 wins** (two rounds, 130 valid rows: 84 animated, 46 live action). On
  live action it finds 98% of the cuts at p = 0.3 (92% at 0.5), where scdet at sptenc's T = 10
  finds 25%, at about the same precision (0.84 against 0.89). On animation it finds more than
  scdet (0.90 at 0.3, 0.87 at 0.5, against 0.84) with fewer false cuts (precision 0.82 and 0.89
  against 0.62: scdet's bursts). Filtering scdet's bursts loses real cuts (recall 0.70). The
  [brief](#decision-brief): TransNetV2 at p = 0.3, no gate, no burst handling, no minimum shot
  length; it misses some cuts inside action anime's bursts (14% of scdet's lone hits there).

- **Three detectors, every frame of 11 sources:** 8 animated (three cartoon episodes, three
  anime episodes, two anime films), two live-action films and a DVD episode, 4.4 hours analysed.
  The detectors are ffmpeg's scdet exactly as sptenc runs it, PySceneDetect's adaptive and
  content detectors, and TransNetV2, a network trained to find shot boundaries. Their candidates,
  merged within ±1 frame, come to 12,798.
- **scdet fires in bursts on action anime, by construction.** Its score is the smaller of a
  frame's mean absolute luma difference to the previous frame (MAFD) and the change of that MAFD,
  so in limited animation every new drawing after a held one scores its whole MAFD. At sptenc's
  threshold T = 10, 28% of its detections on animation come 1–3 frames after the previous one
  (43% on the dark action episode, 37% on the 720p one), and the dark episode gets more than half
  its shots under 0.5 s (782 of 1,472; median 0.33 s). A higher threshold thins the bursts slowly
  (still 23% there at T = 14) while the PySceneDetect cuts it leaves below the threshold grow
  (151 → 267 on that episode).
- **On live action scdet takes fewer than half of what the others find.** On the two films at
  T = 10 it makes 292 detections per hour; 376 more per hour, where both PySceneDetect detectors
  agree, score below 10 (dark and low-contrast cuts, by an earlier look at the sheets).
  TransNetV2 at p = 0.5 takes 302 of those 313 (96%).
- **TransNetV2 hardly bursts:** at p = 0.5, 5 bursts per hour on animation against scdet's 351,
  and 25 shots under 0.5 s per hour against 489. It takes 96% of the animated cuts all detectors
  are sure of (332 of 345, scdet ≥ 30 and both PySceneDetect detectors), but only 10 of the
  grainy cel film's 15.
- **PySceneDetect never bursts, by construction:** its default minimum scene length (15 frames)
  forbids two cuts closer than 13–15 frames, which would also merge real flash cuts (the bright
  anime's 180 shots under 0.5 s at T = 10 looked mostly real on its sheets: text-card flashes).
- **Where they agree:** at working thresholds (TransNetV2 ≥ 0.5, scdet ≥ 10, a PySceneDetect
  detection), all three agree on 2,416 candidates and disagree on 4,148: 1,374 TransNetV2 and one
  other, 239 TransNetV2 alone, 676 the others with TransNetV2 at 0.1–0.5, 1,859 the others with
  TransNetV2 under 0.1 (932 of them lone scdet hits inside anime bursts). The labels that decide
  are on the disagreements ([groups](#where-the-detectors-agree)).
- **Cost** per hour of 1080p Blu-ray, 16 threads, back to back on one episode: decoding alone
  1.96 min, scdet 1.98, PySceneDetect 2.28, TransNetV2 4.30 (half of it the network). Over the
  decode seedvr2x already makes for its frame index, scdet costs about nothing, PySceneDetect
  about 0.3 min and TransNetV2 about 2.3. All three are deterministic.
- **Pipeline findings:** a filter-graph rebuild mid-stream restarts the frame counter sptenc
  reads cut indices from (every cut 14 frames early on the DVD; `-reinit_filter 0` fixes it, and
  seedvr2x counts frames in its own reader); the joins of an earlier split are not all cuts (63 of
  411 score below 10 in a fresh pass, 14 below 4); a PQ-coded HDR film scores far lower (moot while
  v1 refuses HDR).
- **The DVD's labels are out:** its review thumbnails sat 14 frames after their candidates (the
  filter-graph rebuild again, this time restarting the `trim` filter that cuts thumbnails out;
  fixed), and seedvr2x refuses its interlaced source anyway.

## Why it matters for seedvr2x

seedvr2x takes the source itself and cuts it into shots, its processing units
([DESIGN.md](../../seedvr2x/DESIGN.md), "Two kinds of users" and "Pipeline, per shot"); both
workflows start from that detection, the biggest quality lever outside the model. Question 2
measured what each detection error costs against a ground truth
([cuts.md](cuts.md#what-it-means-for-shot-detection)):

- **a missed cut:** 5–31 dB·frames of PSNR-Y over the next shot's first 8 frames on four of six
  cuts, up to 12% of the previous shot showing in its first frame;
- **a false cut:** no fidelity lost; a low-frequency temporal step on a continuous shot, and one
  more processing unit (its first frame encoded alone, its end padded to 4n + 1 frames);
- **a short shot merged into its neighbour:** 10–21 dB·frames over its own frames, from 1 frame
  on: a short shot is better run alone.

So the detector must find every cut first, then not fire in bursts, which chop one shot into many
units. This page compares three detectors on real sources before any label, and sets up the
labels that decide between them.

## Sources

| Source | Kind | What | Analysed | Minutes |
|---|---|---|---|---:|
| cartoon-1, -2, -3 | animation | three web episodes of a bright flat-colour cartoon series ([seeking.md](seeking.md#method)'s S3 and two more): H.264 1080p, 2 Mbit/s | whole, 10,501 frames each | 7.3 each |
| anime-dark | animation | an action anime OVA episode, mostly dark (S8): Blu-ray H.264 1080p | whole, 99,583 frames | 69.2 |
| cel-film | animation | a grainy cel-animated film (S1): Blu-ray H.264 1080p | frames 57,542–93,505 (0:40–1:05) | 25.0 |
| digital-film | animation | a clean digital anime film (S2): Blu-ray H.264 1080p | frames 43,157–79,120 (0:30–0:55) | 25.0 |
| anime-720 | animation | an anime episode kept as the 213 segments of an earlier split, downscaled to 1280×720 | whole, 37,393 frames | 26.0 |
| anime-bright | animation | a bright anime episode with text-card flash cuts (S7): Blu-ray H.264 1080p | whole, 37,327 frames | 25.9 |
| live-1 | live action | an action film with VFX and film grain (S10): Blu-ray H.264 1080p | frames 40,290–76,253 (0:28–0:53) | 25.0 |
| live-2 | live action | a slower film, declared 24 fps but timed at 23.976 (S9): Blu-ray H.264 1080p | frames 43,200–79,199 (0:30–0:55) | 25.0 |
| dvd-sitcom | DVD | a live-action sitcom episode (S6): NTSC MPEG-2 720×480, 29.97 fps interlaced, 16:9 anamorphic | whole, 40,441 frames | 22.5 |

- Frames are numbered from 0 in decode order over the whole source, a list of segments being
  their concatenation; a cut at frame n means a new shot starts at n (scdet and PySceneDetect
  agree on this convention). Every decode is sequential, video only, `-fps_mode passthrough`.
- The films' scdet pass covers the whole film; the statistics and the other detectors cover the
  25-minute chunk.
- anime-dark was scored twice, as its original file and as the 412 segments of an earlier split
  (anime-dark-seg, [below](#pipeline-findings)); the copy is left out of every pooled figure.
- Pooled by kind: animation 3.22 h (8 sources), live action 0.83 h (2), DVD 0.37 h (1).

## Detectors

### scdet, as sptenc runs it

ffmpeg's scdet filter (n9.0.2) scores each frame against the previous one
(`libavfilter/vf_scdet.c:129-131`): MAFD = the mean absolute difference of the luma plane (all
planes for RGB) in percent of the code range, and **score = min(MAFD, |MAFD − previous MAFD|)**,
a cut when it reaches the threshold. sptenc runs it as
`setpts=PTS-STARTPTS,scdet=t=10,metadata=mode=print:key=lavfi.scd.time` into a null output and
reads each detection's frame index from the metadata filter's counter; its 5-second minimum
segment length acts later, on its segments, not on the detection.

Our pass runs the same chain with every key printed for every frame (`lavfi.scd.mafd`, `.score`,
`.time`), video only, so that any threshold can be read back. `check` reproduces sptenc's
detections exactly: on cartoon-1, sptenc's own command and our pass give the same 56 frames at
T = 10 with the same printed scores; on a synthetic clip with hard cuts at frames 30, 60 and 90,
exactly those three (scores 24.9, 22.9, 17.8; at most 2.8 elsewhere). Its candidates are its
local maxima from 4 up.

### PySceneDetect

PySceneDetect 0.7.1's AdaptiveDetector (adaptive threshold 3.0, window 2, minimum content value
15) and ContentDetector (threshold 27, HSV weights 1, 1, 1, 0, flash filter "merge"), both with
their defaults, including a **minimum scene length of 15 frames**. They get the frames its OpenCV
backend would (swscale, the matrix and range the stream declares, BT.601 when it declares none),
downscaled as its SceneManager does by default (to 256 px on the longer side). `pysd-check` runs
its own pipeline beside ours: identical cuts on cartoon-1 (55 and 55), an anime segment and the
synthetic clip, and identical frames, pixel for pixel. Its cuts fall on scdet's frame (offset 0)
on 777 of the 780 scdet peaks ≥ 30 they share; the other 3, a frame later, are on the 720p
episode.

### TransNetV2

TransNetV2 (github.com/soCzech/TransNetV2, commit `85cef72`, MIT licence) is a convolutional
network trained to find shot boundaries. We run its official PyTorch model on the CPU, with the
weights its own `convert_weights.py` converts from the official TensorFlow ones (sha256
`eed5336d…`), on 48×27 RGB frames scaled as the official extraction scales them (ffmpeg's own
scaler), and reproduce its `predict_frames` (windows of 100 frames every 50, each keeping frames
25–74). Every frame gets a probability; a detection is a run of frames at or above p, at its peak.

- **Our port matches the official model:** on 1,000 random frames and on two 3,000-frame
  excerpts (anime-bright, live-1), the official TensorFlow model and ours differ by 5.1 × 10⁻⁷ at
  most in probability, and give the same shots (its `predictions_to_scenes`) and the same
  detections at 0.5 (41 and 35). Our 48×27 frames are byte for byte the official extraction's
  over the first 9,000 and 53,000 frames of those sources.
- **It marks the outgoing shot's last frame:** its peak sits one frame before the cut on 98% of
  the sure cuts that have one (341 of 348; 4 sure cuts have no peak within ±3 frames). Its values
  are shifted by that one frame (`--tnet-offset`, measured per source, −1 everywhere) before any
  comparison.
- Its frames are taken as decoded: an anamorphic source is not unsqueezed, an interlaced one not
  deinterlaced. On the DVD it still finds all 6 sure cuts at p = 0.5.

### Candidates

A candidate is a scdet local maximum ≥ 4, a PySceneDetect detection or a segment join, merged
within ±1 frame (12,137 on the 11 sources), plus TransNetV2's own peaks ≥ 0.1 with none of those
within ±1 frame (661 more: 396 at 0.1–0.3, 99 at 0.3–0.5, 166 from 0.5 up). "Sure" means scdet
≥ 30 with both PySceneDetect detectors within ±1 frame (352 candidates); everything else is
"doubtful". Each candidate carries every detector's value: scdet's score and MAFD, the frames
before and after, its burst count (other frames ≥ 6 within ±6), PySceneDetect's verdicts and
adaptive ratio, TransNetV2's single-frame and all-frames probabilities.

## Statistics (unlabelled)

A **burst** here is a detection 1–3 frames after the previous one; shots are counted between
consecutive detections (a chunk's partial first and last shots left out); "PySceneDetect below
T" counts the cuts both PySceneDetect detectors find where scdet scores below T: possible misses,
until labelled.

### Per hour, by kind

Per hour of source, each cell: detections / bursts / shots under 0.5 s, and for scdet the
PySceneDetect cuts it leaves below T.

| Detector | Animation (3.22 h) | Live action (0.83 h) | DVD (0.37 h) |
|---|---|---|---|
| scdet T = 8 | 1,651 / 625 / 823 / 63 | 523 / 17 / 32 / 217 | 1,150 / 51 / 141 / 5 |
| scdet T = 10 (sptenc) | 1,256 / 351 / 489 / 100 | 292 / 11 / 22 / 376 | 1,110 / 32 / 107 / 11 |
| scdet T = 12 | 982 / 192 / 308 / 159 | 191 / 4 / 12 / 436 | 1,078 / 27 / 99 / 35 |
| scdet T = 14 | 797 / 107 / 195 / 214 | 143 / 2 / 10 / 466 | 960 / 19 / 88 / 133 |
| scdet T = 20 | 451 / 24 / 65 / 393 | 17 / 1 / 1 / 572 | 285 / 19 / 48 / 760 |
| PySceneDetect adaptive | 939 / 0 / 0 | 678 / 0 / 0 | 1,038 / 0 / 13 |
| PySceneDetect content | 889 / 0 / 0 | 840 / 0 / 0 | 1,083 / 0 / 3 |
| TransNetV2 p = 0.3 | 982 / 19 / 67 | 1,010 / 4 / 22 | 1,139 / 19 / 101 |
| TransNetV2 p = 0.5 | 880 / 5 / 25 | 935 / 0 / 7 | 1,094 / 11 / 67 |
| TransNetV2 p = 0.7 | 792 / 3 / 7 | 852 / 0 / 1 | 1,043 / 8 / 29 |
| TransNetV2 p = 0.9 | 617 / 0 / 1 | 431 / 0 / 0 | 928 / 0 / 3 |

- **Animation:** scdet's detections fall by 37% from T = 10 to 14 and its bursts by 70%, but
  the cuts both PySceneDetect detectors find below T double (100 → 214 per hour).
  TransNetV2 at 0.5 detects about what PySceneDetect does, with almost no burst.
- **Live action:** no threshold makes scdet agree with the others: from T = 8 it already leaves
  217 PySceneDetect cuts per hour below it, and TransNetV2 at 0.5 detects three times what scdet
  does at 10. Bursts are rare for every detector.
- **The DVD sitcom:** the detectors' counts agree within 7% (hard cuts, studio lighting); shots
  under 0.5 s per hour: scdet 107 at T = 10, TransNetV2 67 at 0.5, PySceneDetect 3–13.

### Per source

At the working thresholds. scdet and TransNetV2: detections / bursts / shots under 0.5 s.

| Source | scdet T = 10 | PySceneDetect adaptive, content | TransNetV2 p = 0.5 | Sure cuts TransNetV2 takes | PySceneDetect cuts below T = 10: TransNetV2 takes |
|---|---|---|---|---|---|
| cartoon-1 | 56 / 0 / 1 | 55, 55 | 56 / 0 / 1 | 1 of 1 | 0 of 0 |
| cartoon-2 | 61 / 0 / 1 | 60, 60 | 59 / 0 / 1 | 1 of 1 | 0 of 0 |
| cartoon-3 | 100 / 0 / 1 | 99, 108 | 99 / 0 / 1 | 1 of 1 | 3 of 3 |
| anime-dark | 1,473 / 627 / 782 | 1,021, 892 | 818 / 5 / 18 | 77 of 79 | 115 of 151 |
| cel-film | 381 / 67 / 84 | 395, 452 | 410 / 0 / 1 | 10 of 15 | 95 of 97 |
| digital-film | 597 / 101 / 151 | 452, 450 | 457 / 0 / 5 | 48 of 49 | 10 of 13 |
| anime-720 | 717 / 266 / 374 | 451, 379 | 417 / 8 / 17 | 24 of 28 | 49 of 53 |
| anime-bright | 655 / 68 / 180 | 487, 464 | 516 / 3 / 36 | 170 of 171 | 4 of 5 |
| live-1 | 142 / 9 / 18 | 290, 473 | 460 / 0 / 5 | 1 of 1 | 185 of 191 |
| live-2 | 101 / 0 / 0 | 275, 227 | 319 / 0 / 1 | none | 117 of 122 |
| dvd-sitcom | 416 / 12 / 40 | 389, 406 | 410 / 4 / 25 | 6 of 6 | 2 of 4 |

- **The cartoon is clean for everyone:** the three detectors' counts agree within a few, without
  bursts.
- **scdet's bursts follow the animation style:** the action episodes (anime-dark, anime-720)
  burst most, at intervals of 2 frames above all (283 of anime-dark's 627: drawings on twos).
  Right after a held frame (MAFD < 0.5 before it) come 36% of anime-dark's detections at T = 10
  and 71–89% of the cartoon's, but only 6% of the grainy cel film's, which still bursts (18%):
  grain keeps a held drawing's MAFD above 0.5, and a new drawing still steps far above it.
- **TransNetV2's weak spot is the grainy cel film:** 5 of its 15 sure cuts stay under 0.5 (3
  without any peak within ±3 frames), though it takes 95 of the 97 cuts PySceneDetect finds below
  scdet's threshold there.

### Where the detectors agree

Candidates by agreement at the working thresholds (TN = TransNetV2 ≥ 0.5, S = scdet ≥ 10, P = a
detection of either PySceneDetect detector). These groups set round 1's sampling
([below](#round-1-targeted-on-the-disagreements)); the last column counts the rows the first
review already holds.

| Group | Animation | Live action | DVD | All | First-review rows |
|---|---:|---:|---:|---:|---:|
| G1: TN, S and both P | 1,884 | 172 | 360 | 2,416 | 456 |
| G2: TN and S or P, not G1 | 795 | 538 | 41 | 1,374 | 228 |
| G3: TN alone | 160 | 69 | 10 | 239 | 70 |
| G4: S or P, TN at 0.1–0.5 | 603 | 52 | 21 | 676 | 71 |
| G5: S or P, TN under 0.1 | 1,780 | 44 | 35 | 1,859 | 152 |
| … of which scdet alone in a burst | 932 | 2 | 1 | 935 | |
| G6: none of them (scdet 4–10, TN 0.1–0.5, joins) | 5,974 | 160 | 100 | 6,234 | 629 |

- **G1** should be cuts (round 1 checks it on a sample); **G6** holds the low scores kept to
  find what everyone misses.
- **G2 and G3** are TransNetV2's case: on live action they are most of the candidates (607 of
  1,035), the quiet cuts scdet leaves below its threshold.
- **G5** is scdet's (and sometimes PySceneDetect's) case: 932 of its animated candidates are lone
  scdet hits inside bursts, neither TransNetV2 nor PySceneDetect near.
- 151 of TransNetV2's 661 own candidates lie 2–3 frames from an older candidate: the same event
  seen a frame or two apart, or a burst; the labels will tell.

## Cost per detector

Measured back to back on one 1080p Blu-ray episode (anime-bright: H.264 High, 26.3 Mbit/s,
37,327 frames, 25.9 min), 16 threads of the 48-core machine, page cache warm, nothing else
running, two runs in opposite orders:

| Pass | Wall, run 1 / 2 | Per hour of source | CPU time per hour |
|---|---|---:|---:|
| decode alone (video stream, null output) | 50.8 / 50.8 s | 1.96 min | 17.3 min |
| scdet, sptenc's chain (video only) | 51.4 / 51.6 s | 1.98 min | 17.7 min |
| scdet, every key printed (our pass) | 51.3 / 51.8 s | 1.99 min | 17.8 min |
| PySceneDetect, both detectors | 59.4 / 58.8 s | 2.28 min | 26.0 min |
| TransNetV2 | 110.9 / 112.2 s | 4.30 min | 58.4 min |
| … its decode to 48×27 + inference | 52.4 + 57.7 / 52.5 + 59.0 s | 2.02 + 2.25 min | |

- **The decode is the cost.** scdet adds 1% to decoding the frames, PySceneDetect 16% (its
  256-px frames and two detectors); TransNetV2 doubles it, its inference taking as long as the
  decode: 2.25 min per hour on 16 threads, whatever the source's size, since it sees 48×27 frames.
- **Over the decode seedvr2x already runs** (its first pass decodes every frame for the frame
  index, DESIGN.md Input), scdet is a filter in the same graph and costs about nothing,
  PySceneDetect about 0.3 min per hour on 256-px frames from that decode, TransNetV2 about 2.3 on
  48×27 ones. At 4K the decode grows (a UHD HEVC remux decodes at 204 fps on 16 threads, 7 min
  per hour of source) and the detectors' own cost doesn't.
- The official TensorFlow model ran the check's excerpts 1.3–1.4 times faster than the PyTorch
  port (3.4–3.5 s against 4.6–4.9 s per 3,000 frames).
- **Deterministic:** the score pass, PySceneDetect and TransNetV2, each run three times on this
  source (the stored pass and both runs), gave the same scores file byte for byte, the same
  cuts and the same probabilities to the bit.
- DESIGN.md's earlier figures (scdet 1.8, PySceneDetect 2.4, TransNetV2 4.4 min per hour) came
  from the detection passes themselves, on a machine loaded by other jobs; these agree within
  10%.

## Pipeline findings

- **A filter-graph rebuild restarts sptenc's frame count.** The DVD's colour description appears
  at frame 14; ffmpeg then rebuilds the filter graph, which restarts the metadata filter's frame
  counter (and setpts' start and scdet) and reopens its print file. sptenc's command, which reads
  cut indices from that counter, then numbers every later frame from 0 again: all 416 of its
  detections at T = 10 come out 14 frames early, with the same scores. `-reinit_filter 0` keeps one
  graph for every frame (indices right, scores identical). seedvr2x counts frames in its own
  reader, as [seeking.md](seeking.md#failure-mechanisms) recommends for every frame index, never
  from an ffmpeg counter.
- **An earlier split's joins are not all cuts.** anime-dark scored as the 412 segments of an
  earlier split gives the same score on every frame as the original file (largest difference 0,
  the same detections at every threshold; the segments hold 2 more frames at the start). Of its
  411 joins, 63 score below 10 in that fresh pass and 14 below 4 (the lowest 0.016): a join can
  sit inside a continuous shot. DESIGN.md runs the detector inside segments and at their joins.
- **PQ-coded HDR scores low.** scdet's score is a share of the code range, and PQ puts an SDR
  picture's levels in the lower half of it: at T = 10, scdet found 92 shots in the first 70
  minutes of the 4K film of the full-reference clips ([numerics.md](numerics.md#clips)), about
  80 per hour, where PySceneDetect finds 680–840 per hour on the two films here. Moot while
  seedvr2x refuses HDR (DESIGN.md, v1); any threshold would need its own calibration for PQ or
  HLG sources.
- **sptenc's command maps no stream,** so ffmpeg decodes the audio track as well (a Blu-ray's
  lossless audio included): time spent, scores unchanged. seedvr2x reads the video stream alone.
- **The review's thumbnails for the DVD sat 14 frames late.** The sheets cut thumbnails out of a
  sequential decode with ffmpeg's `trim` filter, whose frame count restarts with a rebuilt filter
  graph: every DVD thumbnail showed the frames 14 after its candidate (checked by decoding around
  two rows both ways). The first review's own alignment check had flagged it (0 of 1 sure cut
  aligned). `scd_review.py` now keeps one graph wherever the score pass did. Any frame count
  ffmpeg keeps restarts with a rebuilt graph.
- **TransNetV2's official extraction** (`-f rawvideo`, no frame-rate option) leaves ffmpeg free to
  drop or repeat frames by timestamp; ours passes every decoded frame through (`-fps_mode
  passthrough`), and gives the official bytes on the checked excerpts.

## The review

### The first review: strata and weights

Sessions 1 and 2 made review sheets for every source: 1,734 rows on 125 pages (12 sources with
anime-dark-seg, which was reviewed before its original arrived), each row the frames c − 2, c − 1,
c, c + 1 of one candidate (a real cut falls between c − 1 and c, marked red), with every
detector's values, 15 rows per page. Labels: cut, flash, pan, fade, dissolve, other, not-a-cut.

- **Selection, per source:** a random 10% of the sure candidates (48 rows, to check the
  auto-class); every doubtful candidate up to 120; above that, a stratified sample of 120: the
  decision group first (scdet 6–20, plus PySceneDetect detections and joins scoring below 6: the
  possible misses), then the rest (scdet 4–6 that PySceneDetect ignores, scdet above 20), at
  least 10% of it.
- **Strata:** group × scdet band (4, 6, 8, 10, 12, 14, 17, 20, 30) × PySceneDetect's agreement
  (both, one, none), samples in proportion to stratum size, one row per stratum at least,
  seeded. TransNetV2's own candidates came later as strata of their own, by probability band
  (0.1–0.3 … ≥ 0.9), up to 30 per source (258 rows).
- **Weights:** each row carries its stratum's size and its weight, the candidates it stands for;
  per source they add up to the candidate count. `scd_scores.py summary --labels` turns labels
  into estimates even when only some rows are labelled: per detector and threshold (scdet
  T = 6–20, PySceneDetect's two detectors, TransNetV2 p = 0.1–0.9), the cuts detected and
  missed, recall, false cuts by label, precision with every other label an error and with fades
  and dissolves left out (gradual transitions, where a detector may rightly cut), per source and
  pooled by kind. Recall is relative to the cuts among the candidates.
- The index is only ever appended to: labels written in it are kept byte for byte. No row is
  labelled yet.

### Round 1: targeted on the disagreements

Built on 2026-10-05 (`scd_review.py round`); the labels are under way. Labelling 1,734 rows takes
uninterrupted hours, and most of them sit where the detectors agree; what decides between detectors
is where they don't. So round 1 takes 100 of the first review's rows, by agreement group and kind:

| Group | Animation | Live action | DVD | Why |
|---|---:|---:|---:|---|
| G1, all three | 4 | 2 | 2 | does agreement mean a cut? |
| G2, TransNetV2 and one other | 4 | 6 | 2 | the same, where one of the others is missing |
| G3, TransNetV2 alone | 12 | 12 | 6 | its extra cuts: real, or flashes and motion? |
| G4, the others with TransNetV2 at 0.1–0.5 | 10 | 6 | 4 | what TransNetV2 would miss at 0.5 |
| G5, the others with TransNetV2 under 0.1 | 6, and 6 lone scdet hits in bursts | 4 | 4 | scdet's case |
| G6, none at the working thresholds | 4 | 4 | 2 | what everyone misses |

- **Drawn from the first review's rows** of each cell, with a probability proportional to their
  weight (the candidates each stands for), so that each row drawn stands for (candidates of its
  cell) / (rows drawn): from 1.7 candidates (TransNetV2 alone on the DVD) to 1,494 (G6 on
  animation).
- **Shown blind:** shuffled and numbered on 10 pages of 10 rows, each row's four thumbnails cropped
  from its review page, without any detector's value. One letter per row in a text file: c cut,
  f flash, p pan or motion, d fade or dissolve, o other, n nothing, ? can't tell.
- **Estimates:** `scd_scores.py round` gives, per group and kind, the labels and the share of cuts
  (with its 90% Wilson interval), then the first review's per-detector table (recall, false cuts
  by label, precision), each labelled row standing for its cell's candidates / the cell's
  labelled rows, per kind and for every kind.
- A second round of at most 50 rows if the intervals are too wide to choose (below). The first
  review's index stays as it is; labels written there add to the estimates.

### Round 2: the threshold, a picture-change gate and the bursts

Built and labelled on 2026-10-05 (`scd_review.py round --plan 2`). Round 1 left three
questions open on animation: where TransNetV2's threshold goes (between 0.3 and 0.5 sat 7 cuts
and 3 pans), whether its false cuts on still pictures can be gated away (round 1's cuts all had a
MAFD of 3.35 or more, 8 of its false cuts 0.4–3.0), and how many cuts hide in the bursts it misses
(2 of 6 lone scdet hits there). So round 2 takes 50 more of the first review's rows (none of
round 1's, none of the DVD's), in refined cells: TransNetV2's band (≥ 0.5, 0.3–0.5, 0.1–0.3, under
0.1) and, where it alone fires, the picture's change (MAFD under 4, still, or not):

| Cell | Animation | Live action | Question |
|---|---:|---:|---|
| R3z, TransNetV2 ≥ 0.5 alone, still picture | 6 | 3 | does a gate lose cuts? |
| R3m, TransNetV2 ≥ 0.5 alone, moving | 3 | 2 | the same, moving |
| R4h, TransNetV2 0.3–0.5 with another detector | 6 | 2 | the threshold |
| R6hz, R6hm, TransNetV2 0.3–0.5 alone, still / moving | 3, 3 | 2, 2 | the threshold, and the gate |
| R4l, TransNetV2 0.1–0.3 with another detector | 4 | 1 | below 0.3 |
| R5b, lone scdet hits inside bursts | 8 | | TransNetV2's blind spot |
| R6z, R5: none, or the others alone, TransNetV2 under 0.1 | 3, 2 | | what TransNetV2 misses |

The rows are drawn and shown as round 1's. `scd_scores.py round ROUND1 ROUND2 --dirs ...` pools
both rounds: every labelled row goes to its refined cell (round 1's re-sorted by their own values)
and stands for the cell's candidates / the cell's labelled rows, the cells sized in the episodes'
candidates.csv. Its detectors include TransNetV2 gated on MAFD ≥ 2 and ≥ 3.

## Labelled estimates

Round 1, labelled by the user on 2026-10-05: 99 of the 100 rows (one "can't tell").

- **The DVD's 20 rows are left out.** Its thumbnails show the frames 14 after their candidates:
  the review cut them out of a decode with ffmpeg's `trim` filter, whose frame count restarts
  when the filter graph is rebuilt at frame 14 ([pipeline findings](#pipeline-findings)). Decoded
  around its two G1 rows, the biggest change sits 14 frames early in the default decode and on
  the mark with `-reinit_filter 0`. The first review's own alignment check had flagged it (0 of 1
  sure cut aligned), and seedvr2x refuses the interlaced source anyway (DESIGN.md, Input). Every
  other source passed that check; live-2, which has no sure cut to check, decodes aligned both
  ways.
- That leaves **80 rows: 46 animated, 34 live action**, each standing for its cell's candidates
  / the cell's labelled rows.

Labels by agreement group: cuts / labelled rows (the cell's candidates):

| Group | Animation | Live action |
|---|---|---|
| G1, all three | 4 / 4 (1,884) | 2 / 2 (172) |
| G2, TransNetV2 and one other | 3 / 4 (795) | 5 / 6 (538) |
| G3, TransNetV2 alone | 2 / 12 (160): 7 nothing, 1 flash, 1 pan, 1 fade | 8 / 12 (69): 2 fades, 1 flash, 1 pan |
| G4, the others, TransNetV2 at 0.1–0.5 | 5 / 10 (603) | 3 / 6 (52) |
| G5, the others, TransNetV2 under 0.1 | 0 / 6 (848) | 1 / 4 (42) |
| G5b, lone scdet hits in bursts | 2 / 6 (932) | |
| G6, none at the working thresholds | 0 / 4 (5,974) | 0 / 4 (160) |

Per detector: recall and precision (every label but cut counted as an error), with 90% intervals
from 2,000 bootstraps of the labelled rows within their cells:

| Detector | Animation: recall | precision | Live action: recall | precision |
|---|---|---|---|---|
| scdet ≥ 8 | 0.99 (0.98–1.00) | 0.50 (0.35–0.73) | 0.50 (0.28–0.75) | 0.93 (0.85–0.98) |
| scdet ≥ 10 (sptenc) | 0.84 (0.74–0.95) | 0.69 (0.59–0.79) | 0.24 (0.21–0.32) | 0.86 (0.78–0.95) |
| scdet ≥ 14 | 0.78 (0.66–0.90) | 0.90 (0.80–0.98) | 0.12 (0.00–0.28) | 0.89 |
| scdet ≥ 10 outside bursts | 0.64 (0.56–0.75) | 1.00 | 0.24 (0.21–0.32) | 0.95 (0.87–1.00) |
| PySceneDetect adaptive | 0.87 (0.78–0.97) | 0.78 (0.68–0.88) | 0.76 (0.55–0.91) | 0.82 (0.59–0.97) |
| PySceneDetect content | 0.77 (0.66–0.89) | 0.89 (0.81–1.00) | 0.93 (0.91–0.96) | 0.86 (0.63–0.99) |
| TransNetV2 ≥ 0.3 | 0.88 (0.79–0.98) | 0.84 (0.73–0.93) | 0.99 (0.96–1.00) | 0.86 (0.64–0.99) |
| TransNetV2 ≥ 0.5 | 0.80 (0.72–0.90) | 0.88 (0.76–0.96) | 0.95 (0.91–0.98) | 0.86 (0.63–0.99) |
| TransNetV2 ≥ 0.7 | 0.80 (0.71–0.89) | 0.97 (0.96–0.99) | 0.93 (0.89–0.97) | 0.87 (0.63–0.99) |
| TransNetV2 ≥ 0.5 or scdet ≥ 10 | 0.98 (0.94–1.00) | 0.67 (0.56–0.77) | 0.95 (0.91–0.98) | 0.83 (0.61–0.96) |
| TransNetV2 ≥ 0.3 or PySceneDetect content | 0.88 (0.79–0.98) | 0.78 (0.67–0.88) | 1.00 | 0.84 (0.64–0.97) |

- **Live action: TransNetV2.** At 0.5 it finds 95% of the cuts and scdet at 10 a quarter
  (difference +0.70, 90% interval +0.61 to +0.75), at the same precision; at 0.3, 99%.
  PySceneDetect's content detector comes close (0.93). No scdet threshold gets there: at 8 it
  finds half.
- **Animation: TransNetV2 finds as many cuts as scdet, with fewer false ones.** At 0.5, recall
  0.80 against 0.84 (−0.04, −0.19 to +0.12) and precision 0.88 against 0.69 (+0.20, +0.05 to
  +0.33); at 0.3, recall 0.88 and precision 0.84. scdet's lost precision is its bursts.
- **Between 0.3 and 0.5** sit 7 cuts (4 animated, 3 live action) and 3 pans of the round: 0.3
  gains 7 real cuts for 3 false ones.
- **TransNetV2's misses:** 7 of the 11 cuts it misses at 0.5 score 0.35–0.49; the other 4 score
  0.002–0.17, three of them in action anime where scdet fires in bursts (2 of the 6 lone scdet
  hits inside bursts are real cuts).
- **TransNetV2's false cuts on animation** fall mostly where nothing changes: 7 of its 12 lone
  animated detections are labelled nothing, with scdet at 0.0–1.3 there (held frames, rolling
  credits).
- **A filter on scdet's bursts costs real cuts:** scdet ≥ 10 outside bursts keeps only cuts but
  finds 64% of them on animation. A detector that doesn't burst is the fix.
- **Rows 008 and 010** (a light or white frame, then the new shot at c + 1; labelled other and
  flash) read as cuts: TransNetV2 at 0.5 goes from 0.80 to 0.77 on animation and from 0.95 to
  0.94 on live action. No conclusion changes.
- **The intervals are optimistic:** a bootstrap within cells gives no width to a cell whose few
  rows agree (G6 on animation: 0 cuts in 4 rows standing for 5,974 candidates, whose 90% Wilson
  interval goes up to 0.40). Recall is relative to the cuts among the candidates.

### Rounds 1 and 2 pooled

130 rows (84 animated, 46 live action), each in its refined cell; cuts / labelled rows, the
cell's candidates in brackets:

| Cell | Animation | Live action |
|---|---|---|
| R1, all three | 4 / 4 (1,884) | 2 / 2 (172) |
| R2, TransNetV2 ≥ 0.5 and another | 3 / 4 (795) | 5 / 6 (538) |
| R3m, TransNetV2 ≥ 0.5 alone, moving | 3 / 7 (73) | 8 / 10 (37) |
| R3z, TransNetV2 ≥ 0.5 alone, still (MAFD < 4) | 1 / 14 (87) | 4 / 7 (32) |
| R4h, TransNetV2 0.3–0.5 and another | 6 / 13 (211) | 5 / 5 (31) |
| R6hm, R6hz, TransNetV2 0.3–0.5 alone | 0 / 6 (133) | 1 / 4 (32) |
| R4l, TransNetV2 0.1–0.3 and another | 1 / 7 (392) | 1 / 4 (21) |
| R5, the others, TransNetV2 under 0.1 | 1 / 8 (848) | 1 / 4 (42) |
| R5b, lone scdet hits inside bursts | 2 / 14 (932) | |
| R6l, TransNetV2 0.1–0.3 alone | no row (548) | 0 / 3 (74) |
| R6z, none | 0 / 7 (5,293) | 0 / 1 (54) |

Per detector, recall and precision (90% bootstrap intervals within the cells):

| Detector | Animation: recall | precision | Live action: recall | precision |
|---|---|---|---|---|
| scdet ≥ 10 (sptenc) | 0.84 (0.74–0.97) | 0.62 (0.55–0.69) | 0.25 (0.21–0.32) | 0.89 (0.82–0.97) |
| PySceneDetect content | 0.84 (0.73–0.94) | 0.85 (0.78–0.93) | 0.91 (0.88–0.94) | 0.86 (0.63–0.99) |
| TransNetV2 ≥ 0.3 | 0.90 (0.82–0.98) | 0.82 (0.71–0.89) | 0.98 (0.95–1.00) | 0.84 (0.64–0.97) |
| TransNetV2 ≥ 0.5 | 0.87 (0.79–0.94) | 0.89 (0.75–0.96) | 0.92 (0.88–0.95) | 0.86 (0.63–0.98) |
| TransNetV2 ≥ 0.3, MAFD ≥ 2 | 0.90 (0.82–0.98) | 0.84 (0.73–0.92) | 0.98 (0.95–1.00) | 0.87 (0.65–0.98) |
| TransNetV2 ≥ 0.3, MAFD ≥ 3 | 0.90 (0.82–0.97) | 0.85 (0.73–0.92) | 0.96 (0.92–0.99) | 0.86 (0.65–0.98) |
| scdet ≥ 10 outside bursts | 0.70 (0.63–0.80) | 0.99 | 0.25 (0.21–0.32) | 0.97 (0.94–1.00) |
| TransNetV2 ≥ 0.5 or scdet ≥ 10 | 0.99 (0.98–1.00) | 0.61 (0.53–0.69) | 0.93 (0.89–0.96) | 0.84 (0.62–0.96) |

- **The threshold: 0.3.** Against 0.5 it misses an estimated 92 cuts per hour of animation
  instead of 122, and 19 per hour of live action instead of 70, for more false cuts (precision
  0.82 against 0.89 on animation, 0.84 against 0.86 on live action), which cost no fidelity.
- **No gate.** On the labels, a gate on MAFD ≥ 2 at the cut frame loses no cut and raises
  precision by 0.02–0.03; ≥ 3 already loses live-action cuts. The lowest-MAFD real cuts are a slow
  film's low-contrast ones (2.14, 2.93, 3.35–4.09; and 2.45 on rolling credits), TransNetV2's
  false cuts on still pictures run 0.42–3.87 (11 of 20 under 2): the margin at 2 is 0.14 on 46
  live-action rows. A small precision gain is not worth a cut lost.
- **The blind spot:** 2 of the 14 lone scdet hits inside bursts are real cuts (14%, 90% Wilson
  5–35%), all others flashes, motion or nothing. Those hits come at 433–512 per hour on the
  action anime, 150–190 on the films and the bright anime, none on the cartoon: TransNetV2
  misses about 60–70 cuts per hour of action anime there (25–180 at the interval's ends). Taking
  them back with scdet brings 6 false cuts for each real one.
- **Left out:** animation's 548 candidates where TransNetV2 alone scores 0.1–0.3 got no row in
  either round. At the neighbouring cells' cut rates (0–14%) they would hold at most about 80
  cuts, under 3% of animation's estimated 2,910, which no detector takes at its working
  threshold: every detector's recall would drop by the same share.
- **Borderline rows:** five round-2 rows the user marked borderline (zooms, focus pulls, an
  effect transition; labelled pan or other) read as cuts move TransNetV2 at 0.3 from 0.90 to 0.86
  on animation (three of them score 0.39–0.49, which 0.3 takes and 0.5 misses), scdet from 0.84
  to 0.85; no conclusion changes.

## Decision brief

Joint with question 2 ([cuts.md](cuts.md#what-it-means-for-shot-detection)), 2026-10-05.

Addendum after round 2, 2026-10-05: the threshold stays 0.3, no gate, and nothing beyond the cut
list for fast action anime ([pooled estimates](#rounds-1-and-2-pooled)).

- **Recommendation:** seedvr2x detects shots with TransNetV2 at p = 0.3: the official model
  (MIT; weights converted from the official TensorFlow ones), on 48×27 frames scaled as its
  official extraction scales them, a cut on the frame after its peak (it marks the outgoing
  shot's last frame). No burst handling and no minimum shot length: every detection starts a
  shot, however short. Neither scdet nor PySceneDetect is needed.
- **Evidence:**
  - Live action: recall 0.99 at 0.3 (0.95 at 0.5) against 0.24 for scdet at sptenc's T = 10,
    precision 0.86 for both. scdet at 10 would miss about three quarters of the cuts (an
    estimated 640 per hour on these two films), and question 2 measured a miss at 5–31
    dB·frames on four of its six cuts.
  - Animation: recall 0.88 at 0.3 (0.80 at 0.5) against scdet's 0.84, precision 0.84 (0.88)
    against 0.69. Unlabelled: TransNetV2 at 0.3 bursts 19 times per hour and makes 67 shots
    under 0.5 s per hour; scdet at 10, 351 and 489.
  - 0.3 over 0.5: 7 more real cuts for 3 more false ones (pans) in the round. A miss costs
    fidelity, a false cut none (cuts.md: a low-frequency step on a continuous shot at most).
  - Bursts: TransNetV2 needs no filter; filtering scdet's bursts loses real cuts (recall 0.64).
  - Minimum shot length: none. Question 2 found a short shot better run alone than merged, from
    1 frame on (10–21 dB·frames), and real cuts come 1–3 frames apart in action anime.
  - Cost: about 2.3 min of CPU per hour of 1080p source over the decode the frame index already
    makes (16 threads), deterministic; probably far less on the GPU (not measured).
- **Confidence:** high for live action, for leaving scdet as sptenc runs it, and for 0.3 over 0.5
  (28 labelled rows in TransNetV2's 0.3–0.5 band, 18 of them round 2's); medium for the animated
  recall's level
  (130 rows, some standing for up to 760 candidates; 548 candidates in a cell no round drew).
- **Caveats:**
  - TransNetV2's blind spot is fast action anime, where scdet bursts: 2 of 6 lone scdet hits
    there were real cuts it scores under 0.1. A union with scdet takes them back with the bursts
    (recall 0.98, precision 0.67 on animation).
  - Its false cuts on animation fall where nothing changes (held frames, rolling credits; scdet
    0.0–1.3): a gate on a picture change at the candidate would likely remove them (untested).
  - Its inputs must be scaled as the official extraction scales them (ffmpeg's default scaler to
    48×27); another scaler is untested.
  - The DVD is out (its labels were on misaligned thumbnails; refused as interlaced). Two
    live-action films; no broadcast or clean digital live action.
  - Round 2 (50 more rows): 0.3 confirmed; a picture-change gate (MAFD ≥ 2 at the cut frame)
    loses no labelled cut but gains only 0.02–0.03 precision with a 0.14 margin to a slow film's
    low-contrast cuts: not recommended; the blind spot is 14% (5–35%) of scdet's lone hits in
    bursts, about 60–70 cuts per hour of action anime, not worth scdet's bursts back.

## Native 4K animation: cuts inside fast camera motion

2026-10-06, while choosing 4K full-reference clips ([numerics.md](numerics.md#clips)) from *Sol
Levante* (Netflix / Production I.G, CC BY 4.0), a short made natively in 4K: its SDR master,
6,314 frames at 24 fps, run whole through scdet and TransNetV2. TransNetV2 makes 61 detections at
0.3, 100 at 0.1.

A fast camera flight (frames 1619–1711) looked like one take to every detector. The user, checking
it in a player, found two real cuts, which frame strips then placed at frames 1639 and 1661. Between
them is a 22-frame shot of a bird flying. Later in the same flight, frames 1686–1692 are one shot:
the bird changes shape as it flies off into the distance, the user confirmed on the extracted
frames.

| Frames | What | TransNetV2 (single, peak) | scdet score |
|---|---|---|---|
| 1639 | real cut | 0.207 | 1.8 |
| 1661 | real cut | 0.155 | 1.3 |
| 1658, 1664 | no cut | 0.161, 0.142 | 0.8, 4.6 |
| 1691 | no cut (one shot) | 0.232 | 0.0–0.9 |

The whole frame moves fast here: the mean absolute difference between consecutive frames is 13–16
on every frame, so a cut between two similar-coloured moving shots is no larger a step than the
motion itself. In this flight no threshold separates the real cuts from the rest:

| Threshold | Real cuts caught | False cuts |
|---|---|---|
| 0.3 (the default) | 0 of 2 | 0 |
| 0.2 | 1 of 2 | 1 |
| 0.15 | 2 of 2 | 2 |

This adds native 4K animation's fast camera work to the blind spot above. The user proposed letting
the threshold be set at the plan stage: a user who sees cuts missing lowers it and checks the cut
list again before the upscale. Listing the near-misses (0.1–0.3) beside the cuts would make that
check quick: 39 of them on this film.

## Caveats

- **Labels on 130 rows.** The statistics count detections; only the estimates rest on labels,
  and those on 130 rows, some of which stand for up to 760 candidates each. The DVD's 20 rows
  are out (misaligned thumbnails).
- **Recall will be relative** to the cuts among the candidates, the union of every detector's at
  low thresholds (scdet local maxima from 4, TransNetV2 peaks from 0.1): a cut no detector comes
  near is never shown.
- **Mostly animation:** 3.22 of the 4.4 hours. Live action is two films (25 minutes each) and a
  DVD episode; no broadcast source, no clean digital live action, no SD anime.
- **TransNetV2's alignment** was measured on sure cuts, all hard; on gradual transitions its peak
  may sit elsewhere.
- **The costs** come from one 1080p H.264 source on 16 threads of a 48-core machine. The
  decode's share changes with codec, bitrate and size (the 2 Mbit/s cartoon decodes several times
  faster, a UHD HEVC remux slower); a consumer CPU has fewer cores.

## Reproduce

```bash
S=scripts/scd_scores.py; R=scripts/scd_review.py; O=out/scd   # EP = one source's directory
# sptenc's own command against the all-score pass; a synthetic clip with cuts at 30, 60, 90
python3 $S check SRC --out $O/check/one
python3 $S check $O/check/synth/synth.mkv --out $O/check/synth --synthetic
# every frame's scdet score: a file, a directory of segments or @LIST; --no-reinit where the
# stream's parameters change mid-way (-reinit_filter 0)
python3 $S score SRC --out $O/EP [--no-reinit]
# PySceneDetect on the same frames (a film's chunk: --range 40:00..1:05:00), and its own pipeline
python3 $S pysd --out $O/EP [--range A..B]
python3 $S pysd-check FILE --out $O/check/one
# TransNetV2: the official checkout and its converted weights; the official TensorFlow model beside
export TNET_DIR=/path/to/TransNetV2 TNET_WEIGHTS=/path/to/transnetv2-pytorch-weights.pth
python3 $S tnet --out $O/EP --threads 16
python3 $S tnet-check FILE --out $O/check/tnet --start 6000 --frames 3000 --random 1000
# candidates, statistics per threshold; an original against its segments; tables (and estimates)
python3 $S analyse --out $O/EP
python3 $S compare $O/anime-dark $O/anime-dark-seg --md compare.md
python3 $S summary $O/EP... --md summary.md [--labels $O/review/index.csv]
# review sheets, TransNetV2's own candidates appended, the combined index
python3 $R sheets $O/EP... --out $O/review
python3 $R extend $O/review $O/EP... --cap 30 --skip anime-dark-seg
python3 $R index $O/review
# round 1: 100 of the review's rows by agreement group and kind, blind; estimates from its labels
python3 $R round $O/review $O/EP... --out $O/round1
python3 $S round $O/round1 --skip dvd-sitcom --md round1.md   # the DVD: thumbnails misaligned
# round 2: 50 more rows in refined cells; then both rounds pooled in those cells
python3 $R round $O/review $O/EP... --out $O/round2 --plan 2 --exclude $O/round1/rows.csv
python3 $S round $O/round1 $O/round2 --dirs $O/EP... --skip dvd-sitcom --md rounds.md
# costs on one source, 16 threads, page cache warm: decode alone, then sptenc's chain
# (score, pysd and tnet as above: each logs its own seconds)
ffmpeg -threads 16 -i SRC -map 0:v:0 -fps_mode passthrough -f null -
ffmpeg -threads 16 -i SRC -map 0:v:0 \
  -vf setpts=PTS-STARTPTS,scdet=t=10,metadata=mode=print:key=lavfi.scd.time -f null -
```
