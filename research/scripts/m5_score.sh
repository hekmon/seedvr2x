#!/bin/bash
# Milestone 5's scoring of m5_run.sh's stage rest, CPU only, niced, from the 16-bit masters
# (seedvr2x/DESIGN.md, Validation milestones 5):
#   bash m5_score.sh PY b        clip B one batch, and clip B on 6:2 windows (ours --window 6
#                                against numz + STITCH_LATENT 6:2, each against its own one-batch run)
#   bash m5_score.sh PY fr CLIP  one full-reference clip (+ ΔE00 lf to the ground truth, fr_metrics.py)
# PY: a python with numpy, cv2 and PIL (numz's venv's).
#
# Historical: the record of the first run's scoring (2026-10-04), kept to re-run it. Both sides:
# numz's masters, and ours as first run (seedvr2x 27ce6ba, lab's bf16 reference), through
# m5_score.py, whose verdict has the rules as first written (4 of 11 materials passed);
# m5_score3b.sh scores step 3b's masters under the rules as sharpened, and reuses the numz JSONs
# this script writes: q-numz-*.json, s-numz-b-stitch.json and fr/<clip>.numz.s42.json.
# Milestone 1's input was scored alike by m5_score.py directly: quality on ours/m1-lab1.mkv and on
# numz_lab.mkv against milestone 1's input (labels ours-m1, numz-m1), psnr into p-m1.json,
# verdict m1. seedvr2x's `--color-correction lab` went with 68b1529, when split replaced lab.
# It calls research/scripts' quality_metrics.py and stitch_metrics.py through m5_score.py, and
# fr_metrics.py (as they were at eca0ff1; fr_metrics.py has changed since in its VMAF models only,
# not in the ΔE00 the verdict reads). Clip B's input: m5_run.sh's header.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

usage() {
    cat >&2 <<'EOF'
usage: m5_score.sh PY b | m5_score.sh PY fr CLIP
Environment, every variable required, paths absolute (the scoring runs in M5_OUT/metrics), PY
and M5_FRPY too unless found on PATH:
  M5_OUT      m5_run.sh's output: numz/, ours/; the scores go to metrics/
  M5_SCRIPTS  research/scripts: quality_metrics.py, stitch_metrics.py, fr_metrics.py
  M5_FRPY     a python with fr_metrics.py's requirements
  M5_B        clip B's input_b.mkv
  M5_FR       the full-reference clips' directory (<clip>.d1.lr.mkv and <clip>.gt.mkv)
ffmpeg and ffprobe on PATH.
EOF
    exit 2
}
case "${2:-}" in b) ;; fr) [ -n "${3:-}" ] || usage ;; *) usage ;; esac
for v in M5_OUT M5_SCRIPTS M5_FRPY M5_B M5_FR; do
    [ -n "${!v:-}" ] || { echo "m5_score.sh: $v is not set" >&2; usage; }
done

PY=$1
# The scoring runs in M5_OUT/metrics (cd below), where a relative path went missing, said only as
# "MISSING out/ours/b-lab1.mkv": refused, a python's too unless found on PATH (no slash).
for v in M5_OUT M5_SCRIPTS M5_B M5_FR PY M5_FRPY; do
    case $v in PY | M5_FRPY) [[ ${!v} == */* ]] || continue ;; esac
    [[ ${!v} == /* ]] || { echo "m5_score.sh: $v is not an absolute path: ${!v}" >&2; usage; }
done

export CUDA_VISIBLE_DEVICES= M5_SCRIPTS
OUT=$M5_OUT
B=$M5_B
FR=$M5_FR
SCORE="nice -n 19 $PY $HERE/m5_score.py"
FRPY=$M5_FRPY
mkdir -p "$OUT/metrics/fr"
cd "$OUT/metrics" || exit 1

quality() {  # label master input
    if [ -f "$2" ]; then
        $SCORE quality "$2" --input "$3" --label "$1" --json "q-$1.json" > /dev/null \
            || echo "FAILED quality $1"
    else
        echo "MISSING $2"
    fi
}

case "$2" in
b)
    echo "== clip B, one batch"
    quality ours-b "$OUT/ours/b-lab1.mkv" "$B"
    quality numz-b "$OUT/numz/b.mkv" "$B"
    quality ours-b-none "$OUT/ours/b-none1.mkv" "$B"
    $SCORE psnr "$OUT/ours/b-lab1.mkv" "$OUT/numz/b.mkv" --json p-b.json
    $SCORE verdict b --ours q-ours-b.json --numz q-numz-b.json --psnr p-b.json
    echo "== clip B, 6:2 windows"
    quality ours-b-w6 "$OUT/ours/b-w6.mkv" "$B"
    quality numz-b-stitch "$OUT/numz/b-stitch.mkv" "$B"
    $SCORE stitch "$OUT/ours/b-w6.mkv" --input "$B" --ref "$OUT/ours/b-lab1.mkv" --latent 6:2 \
        --curve cosine --label ours-b-w6 --json s-ours-b-w6.json > /dev/null \
        || echo "FAILED stitch ours"
    $SCORE stitch "$OUT/numz/b-stitch.mkv" --input "$B" --ref "$OUT/numz/b.mkv" --latent 6:2 \
        --curve cosine --label numz-b-stitch --json s-numz-b-stitch.json > /dev/null \
        || echo "FAILED stitch numz"
    $SCORE psnr "$OUT/ours/b-w6.mkv" "$OUT/numz/b-stitch.mkv" --json p-b-w6.json
    $SCORE verdict b-w6 --ours q-ours-b-w6.json --numz q-numz-b-stitch.json \
        --stitch-ours s-ours-b-w6.json --stitch-numz s-numz-b-stitch.json --psnr p-b-w6.json
    ;;
fr)
    clip=$3
    echo "== $clip"
    input="$FR/$clip.d1.lr.mkv"
    quality "ours-fr-$clip" "$OUT/ours/fr-$clip.mkv" "$input"
    quality "numz-fr-$clip" "$OUT/numz/fr-$clip.mkv" "$input"
    $SCORE psnr "$OUT/ours/fr-$clip.mkv" "$OUT/numz/fr-$clip.mkv" --json "p-fr-$clip.json"
    nice -n 19 $FRPY "$M5_SCRIPTS/fr_metrics.py" "$FR/$clip.gt.mkv" --clip "$clip" \
        --out ours 42 "$OUT/ours/fr-$clip.mkv" --out numz 42 "$OUT/numz/fr-$clip.mkv" \
        --json-dir fr --no-deep --threads 12 --vmaf-threads 8 > "fr-$clip.log" 2>&1 \
        || { echo "FAILED fr_metrics $clip"; tail -n 5 "fr-$clip.log"; }
    tail -n 3 "fr-$clip.log"
    $SCORE verdict "fr-$clip" --ours "q-ours-fr-$clip.json" --numz "q-numz-fr-$clip.json" \
        --fr-ours "fr/$clip.ours.s42.json" --fr-numz "fr/$clip.numz.s42.json" \
        --psnr "p-fr-$clip.json"
    ;;
*)
    usage
    ;;
esac
echo "== scored $2 ${3:-} $(date -u +%FT%TZ)"
