#!/bin/bash
# Milestone 5's step 3b scoring, CPU only and niced: m5_run3b.sh's masters (ours3b/) against
# numz's masters and metrics JSONs (m5_run.sh, m5_score.sh), under milestone 5's rules as
# sharpened on 2026-10-04 (m5_score3b.py's verdict; seedvr2x/DESIGN.md, Validation milestones 5),
# and each master's frames against the ref32 experiment's, when its masters are there:
#   bash m5_score3b.sh PY m1|b|fr CLIP|all
# PY: numz's venv's python (numpy, opencv), as for m5_score.sh.
#
# Historical: the record of step 3b's scoring (2026-10-04), kept to re-run it: 11 of 11
# materials passed, seedvr2x at eca0ff1. Step 3b ran it material by material as each run ended
# (m1, then m5_lab_fig.py on ours3b/m1.mkv; b once b and b-w6 had ended; fr CLIP for each clip),
# which `all` does in one go. ref32 was the experiment before step 3b: lab's reference swapped to
# float32 at run time in seedvr2x 27ce6ba; step 3b's masters equal its frame for frame on the 10
# materials it ran (not b-w6). Its scripts aren't kept here; SAME reads n/a without its masters
# (ref32/).
# numz's quality JSONs are scored again only when they lack the unrounded spreads; numz's
# s-numz-b-stitch.json and fr/<clip>.numz.s42.json must be there (m5_score.sh b, fr CLIP).
# seedvr2x's `--color-correction lab` went with 68b1529, when split replaced lab.
# It calls research/scripts' quality_metrics.py and stitch_metrics.py through m5_score3b.py, and
# fr_metrics.py (as they were at eca0ff1; fr_metrics.py has changed since in its VMAF models only,
# not in the ΔE00 the verdict reads). Clip B's input: m5_run.sh's header.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

usage() {
    cat >&2 <<'EOF'
usage: m5_score3b.sh PY m1|b|fr CLIP|all
Environment, every variable required, paths absolute (the scoring runs in M5_OUT/metrics), PY
and M5_FRPY too unless found on PATH:
  M5_OUT      m5_run.sh's and m5_run3b.sh's output: numz/, ours3b/, metrics/ (ref32/ if any)
  M5_SCRIPTS  research/scripts: quality_metrics.py, stitch_metrics.py, fr_metrics.py
  M5_FRPY     a python with fr_metrics.py's requirements
  M5_M1       milestone 1's input_rgb.mkv; numz's lab master next to it (numz_lab.mkv)
  M5_B        clip B's input_b.mkv
  M5_FR       the full-reference clips' directory (<clip>.d1.lr.mkv and <clip>.gt.mkv)
ffmpeg and ffprobe on PATH.
EOF
    exit 2
}
case "${2:-}" in m1 | b | all) ;; fr) [ -n "${3:-}" ] || usage ;; *) usage ;; esac
for v in M5_OUT M5_SCRIPTS M5_FRPY M5_M1 M5_B M5_FR; do
    [ -n "${!v:-}" ] || { echo "m5_score3b.sh: $v is not set" >&2; usage; }
done

PY=$1
# The scoring runs in M5_OUT/metrics (cd below), where a relative path went missing, said only as
# "MISSING out/ours3b/b.mkv": refused, a python's too unless found on PATH (no slash).
for v in M5_OUT M5_SCRIPTS M5_M1 M5_B M5_FR PY M5_FRPY; do
    case $v in PY | M5_FRPY) [[ ${!v} == */* ]] || continue ;; esac
    [[ ${!v} == /* ]] || { echo "m5_score3b.sh: $v is not an absolute path: ${!v}" >&2; usage; }
done

export CUDA_VISIBLE_DEVICES= M5_SCRIPTS
OUT=$M5_OUT
M1=$M5_M1
NUMZ_M1="$(dirname "$M1")/numz_lab.mkv"
B=$M5_B
FR=$M5_FR
FR_CLIPS="anime-clean anime-grain anime-dark cartoon-bright anime-sky anime-bright live-vfx live-slow"
SCORE="nice -n 19 $PY $HERE/m5_score3b.py"
FRPY=$M5_FRPY
mkdir -p "$OUT/metrics/fr"
cd "$OUT/metrics" || exit 1

frames_md5() {  # the decoded frames' md5, as stored (gbrp16le)
    nice -n 19 ffmpeg -v error -nostdin -i "$1" -map 0:v:0 -f md5 - 2>&1
}

same() {  # name: ours3b's frames against ref32's
    if [ ! -f "$OUT/ref32/$1.mkv" ]; then
        echo "SAME $1 n/a: no ref32 run"
        return
    fi
    local a b
    a=$(frames_md5 "$OUT/ours3b/$1.mkv")
    b=$(frames_md5 "$OUT/ref32/$1.mkv")
    if [ "$a" = "$b" ]; then echo "SAME $1 identical frames ($a)"; else echo "SAME $1 DIFFER: $a against $b"; fi
}

quality() {  # label master input
    if [ -f "$2" ]; then
        $SCORE quality "$2" --input "$3" --label "$1" --json "q-$1.json" > /dev/null \
            || echo "FAILED quality $1"
    else
        echo "MISSING $2"
    fi
}

spreads() {  # label master input: numz's quality JSON again if it lacks the unrounded spreads
    if ! $PY -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if 'lab_std_in' in d and 'lab_std_out' in d else 1)" "q-$1.json"; then
        echo "== q-$1.json lacks lab_std_in/out: scored again"
        quality "$1" "$2" "$3"
    fi
}

m1() {
    echo "== m1 $(date -u +%FT%TZ)"
    same m1
    quality ours3b-m1 "$OUT/ours3b/m1.mkv" "$M1"
    spreads numz-m1 "$NUMZ_M1" "$M1"
    $SCORE psnr "$OUT/ours3b/m1.mkv" "$NUMZ_M1" --json p-ours3b-m1.json
    $SCORE verdict m1 --ours q-ours3b-m1.json --numz q-numz-m1.json --psnr p-ours3b-m1.json
}

b() {
    echo "== clip B, one batch $(date -u +%FT%TZ)"
    same b
    quality ours3b-b "$OUT/ours3b/b.mkv" "$B"
    spreads numz-b "$OUT/numz/b.mkv" "$B"
    $SCORE psnr "$OUT/ours3b/b.mkv" "$OUT/numz/b.mkv" --json p-ours3b-b.json
    $SCORE verdict b --ours q-ours3b-b.json --numz q-numz-b.json --psnr p-ours3b-b.json
    echo "== clip B, 6:2 windows $(date -u +%FT%TZ)"
    same b-w6
    quality ours3b-b-w6 "$OUT/ours3b/b-w6.mkv" "$B"
    spreads numz-b-stitch "$OUT/numz/b-stitch.mkv" "$B"
    $SCORE stitch "$OUT/ours3b/b-w6.mkv" --input "$B" --ref "$OUT/ours3b/b.mkv" --latent 6:2 \
        --curve cosine --label ours3b-b-w6 --json s-ours3b-b-w6.json > /dev/null \
        || echo "FAILED stitch ours3b"
    $SCORE psnr "$OUT/ours3b/b-w6.mkv" "$OUT/numz/b-stitch.mkv" --json p-ours3b-b-w6.json
    $SCORE verdict b-w6 --ours q-ours3b-b-w6.json --numz q-numz-b-stitch.json \
        --stitch-ours s-ours3b-b-w6.json --stitch-numz s-numz-b-stitch.json --psnr p-ours3b-b-w6.json
}

fr() {
    local clip=$1 input="$FR/$1.d1.lr.mkv"
    echo "== $clip $(date -u +%FT%TZ)"
    same "fr-$clip"
    quality "ours3b-fr-$clip" "$OUT/ours3b/fr-$clip.mkv" "$input"
    spreads "numz-fr-$clip" "$OUT/numz/fr-$clip.mkv" "$input"
    $SCORE psnr "$OUT/ours3b/fr-$clip.mkv" "$OUT/numz/fr-$clip.mkv" --json "p-ours3b-fr-$clip.json"
    nice -n 19 $FRPY "$M5_SCRIPTS/fr_metrics.py" "$FR/$clip.gt.mkv" --clip "$clip" \
        --out ours3b 42 "$OUT/ours3b/fr-$clip.mkv" \
        --json-dir fr --no-deep --threads 12 --vmaf-threads 8 > "fr-ours3b-$clip.log" 2>&1 \
        || { echo "FAILED fr_metrics $clip"; tail -n 5 "fr-ours3b-$clip.log"; }
    tail -n 3 "fr-ours3b-$clip.log"
    $SCORE verdict "fr-$clip" --ours "q-ours3b-fr-$clip.json" --numz "q-numz-fr-$clip.json" \
        --fr-ours "fr/$clip.ours3b.s42.json" --fr-numz "fr/$clip.numz.s42.json" \
        --psnr "p-ours3b-fr-$clip.json"
}

case "${2:-}" in
m1) m1 ;;
b) b ;;
fr) fr "$3" ;;
all)
    m1
    b
    for clip in $FR_CLIPS; do fr "$clip"; done
    ;;
*)
    usage
    ;;
esac
echo "== scored3b ${2:-} ${3:-} $(date -u +%FT%TZ)"
