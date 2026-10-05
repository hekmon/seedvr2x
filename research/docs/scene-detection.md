# Scene detection

> Status: **statistics measured, labels pending** (2026-10-05), for DESIGN.md's open question
> on seedvr2x's own scene detection ([DESIGN.md](../../seedvr2x/DESIGN.md#open-questions)):
> which detector, which threshold, what to do with bursts, and whether shots need a minimum
> length. Three detectors ran on every frame of 11 sources; their candidates wait for the user's
> labels, which turn the counts below into recall and false cuts.
> [Labelled estimates](#labelled-estimates) and the [decision brief](#decision-brief), joint with
> question 2 ([cuts.md](cuts.md)), are placeholders until then. Tools:
> [`scripts/scd_scores.py`](../scripts/scd_scores.py) (scores, detectors, candidates,
> statistics, label estimates), [`scripts/scd_review.py`](../scripts/scd_review.py) (review
> sheets and index).

In short (unlabelled: these are detections, not yet errors):

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
- **What the labels decide:** recall and false cuts per detector, threshold and kind, from a
  [round of about 100 rows](#round-1-targeted-on-the-disagreements) aimed at the disagreements
  (when the user is ready), then the joint brief with question 2's costs: the detector, its
  threshold, what to do with bursts, and whether shots need a minimum length.

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
running, two rounds in opposite orders:

| Pass | Wall, round 1 / 2 | Per hour of source | CPU time per hour |
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
  source (the stored pass and both rounds), gave the same scores file byte for byte, the same
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

Planned, for when the user is ready. Labelling 1,734 rows takes uninterrupted hours, and most of
them sit where the detectors agree; what decides between detectors is where they don't. So round 1
draws about 100 rows by agreement group and kind (animation, live action, DVD):

| Group | Rows | Why |
|---|---:|---|
| G3, TransNetV2 alone | 30 | its extra cuts: real, or flashes and motion? |
| G4, scdet or PySceneDetect with TransNetV2 at 0.1–0.5 | 20 | what TransNetV2 would miss at 0.5 |
| G5, scdet or PySceneDetect with TransNetV2 under 0.1 | 20 | scdet's case, about 6 of them inside bursts |
| G1 and G2, TransNetV2 with the others | 20 | does agreement mean a cut? |
| G6, none at the working thresholds | 10 | what everyone misses |

- **Compact sheets:** the same four frames per row, more rows per page, and a batch file with
  one-letter labels (c cut, f flash, p pan, d fade or dissolve, o other, n not a cut).
- **Weights:** each row stands for its group-and-kind cell (cell size / rows drawn), so the
  summary reads the batch like the first review's index. Thresholds other than the working ones
  are estimated from the rows' own values, more coarsely than the first review's score bands
  allow.
- A second round of at most 50 rows if the intervals are too wide to choose. The first review's
  index stays as it is; any labels written there add to the estimates.

## Labelled estimates

*To come with the labels.* Per detector and threshold, by kind: recall, false cuts by label
(flash, pan, fade or dissolve, other, not a cut), bursts that are real cuts, short shots that are
real; whether the sure candidates are all cuts.

## Decision brief

*To come with the labels*, joint with question 2
([cuts.md](cuts.md#what-it-means-for-shot-detection)): the detector (or a combination), its
threshold, what to do with bursts, whether shots need a minimum length, and the cost per hour.

## Caveats

- **Unlabelled.** A burst may be real flash cuts; a PySceneDetect cut below scdet's threshold
  may be a pan, a fade or a flash; a TransNetV2-only candidate may be motion. Every count above
  is a detection until the labels say what it is.
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
# costs on one source, 16 threads, page cache warm: decode alone, then sptenc's chain
# (score, pysd and tnet as above: each logs its own seconds)
ffmpeg -threads 16 -i SRC -map 0:v:0 -fps_mode passthrough -f null -
ffmpeg -threads 16 -i SRC -map 0:v:0 \
  -vf setpts=PTS-STARTPTS,scdet=t=10,metadata=mode=print:key=lavfi.scd.time -f null -
```
