#!/bin/bash
# Milestone 5's GPU runs (seedvr2x/DESIGN.md, Validation milestones 5: seedvr2x's lab against
# numz's lab, one batch = one shot), one stage at a time:
#   m1    numz's lab on milestone 1's input, then ours lab/none/lab/none on it in ONE lock hold
#   rest  clip B (numz's lab, the cost hold, numz + STITCH_LATENT 6:2 with lab, ours --window 6
#         with lab), then each full-reference clip (numz's lab, ours lab)
#
# Historical: the record of what ran on 2026-10-03/04, kept to re-run it. seedvr2x's
# `--color-correction lab` went with 68b1529, when split replaced lab.
# - numz's side, the masters every verdict compares with: numz 4490bd1 through research/scripts'
#   bench.py and ffv1_out.py (16-bit RGB masters), --color_correction lab, one batch per material;
#   clip B in 6:2 windows through blend_patch.py (STITCH_LATENT=6:2, STITCH_CURVE=cosine).
# - ours' side as first run: seedvr2x at 27ce6ba, lab's reference the encoder's bf16 input (4 of
#   11 materials passed). m5_run3b.sh ran ours' side again at eca0ff1, the float32 reference
#   (step 3b, 11 of 11); this script run at eca0ff1 gives step 3b's ours too.
# - lab's cost: lab, none, lab, none back to back in one hold of the GPU lock, on milestone 1's
#   input and on clip B: each unit's time and VRAM peak from seedvr2x's log, the wall time and
#   peak RSS (/usr/bin/time -v), the lab work directory's file sizes sampled every 0.5 s; read by
#   m5_cost.py.
# research/scripts as they were at eca0ff1 (bench.py, ffv1_out.py and blend_patch.py here;
# ffv1_out.py has changed since, in its YUV variants only, not in the 16-bit RGB masters these
# runs write). Then m5_score.sh scores stage rest, m5_cost.py and m5_sizes.py give lab's cost
# and the masters' sizes.
#
# The materials: milestone 1's input (seedvr2x/tests/test_regression.py says how it was made),
# clip B's (below), and the full-reference clips' degraded inputs, research/scripts/fr_clips.py's
# d1 (<clip>.d1.lr.mkv: 960x540 8-bit RGB FFV1, 45 frames).
#
# Clip B's input (M5_B), no test reading it any more: 81 frames of 8-bit RGB FFV1 at 1920x1080,
# frames 20-100 of SEGMENT, a 1920x1080 yuv420p HEVC segment of an animated episode (limited
# range, chroma left, no colour tags, 23.976 fps; research/docs/stitching.md's clip B: dark,
# animated on threes), BT.709 limited range to full-range RGB by zimg, the chroma upsampled by
# zscale's default (bilinear), read alike by numz (OpenCV) and by seedvr2x; made with ffmpeg
# n9.0.2 (with zimg):
#
#   VF=trim=start_frame=20:end_frame=101,setpts=PTS-STARTPTS
#   VF=$VF,zscale=matrixin=709:transferin=709:primariesin=709:rangein=limited:range=full
#   VF=$VF:transfer=709:primaries=709:dither=none,format=gbrp
#   ffmpeg -nostdin -y -i SEGMENT -map 0:v:0 -vf "$VF" -fps_mode passthrough \
#     -c:v ffv1 -level 3 -g 1 -pix_fmt bgr0 -an -sn input_b.mkv
#
# Remade so on 2026-10-09: the same frames (framemd5) and packets, the files differing only in
# the Matroska muxer's random UIDs; one filter thread or many, the same frames.
#
# Every run holds the GPU lock (flock), with the nvidia-smi check inside the hold; a run is
# skipped (STATUS ... skipped) if another compute process shows. Each run ends with a STATUS line.
set -u

usage() {
    cat >&2 <<'EOF'
usage: m5_run.sh m1|rest
Environment, every variable required, paths absolute (the runs cd into the checkouts):
  M5_OUT       where the runs go: numz/, ours/, logs/
  M5_SEEDVR2X  seedvr2x's project directory (its .venv), at the commit to run
  M5_NUMZ      numz's checkout at 4490bd1 (its .venv)
  M5_SCRIPTS   research/scripts: bench.py, ffv1_out.py, blend_patch.py
  M5_MODELS    the model directory: seedvr2_ema_7b_fp16.safetensors and numz's VAE
  M5_RUNS      bench.py's runs directory
  M5_LOCK      the GPU lock file every run holds
  M5_M1        milestone 1's input_rgb.mkv; numz's lab master goes next to it (numz_lab.mkv)
  M5_B         clip B's input_b.mkv
  M5_FR        the full-reference clips' directory (<clip>.d1.lr.mkv)
ffmpeg and ffprobe on PATH: the runs' were n9.0.2 with zimg.
EOF
    exit 2
}
case "${1:-}" in m1 | rest) ;; *) usage ;; esac
for v in M5_OUT M5_SEEDVR2X M5_NUMZ M5_SCRIPTS M5_MODELS M5_RUNS M5_LOCK M5_M1 M5_B M5_FR; do
    [ -n "${!v:-}" ] || { echo "m5_run.sh: $v is not set" >&2; usage; }
    [[ ${!v} == /* ]] || { echo "m5_run.sh: $v is not an absolute path: ${!v}" >&2; usage; }
done

LOCK=$M5_LOCK
OUT=$M5_OUT
SX=$M5_SEEDVR2X
NUMZ=$M5_NUMZ
SCRIPTS=$M5_SCRIPTS
MODELS=$M5_MODELS
RUNS=$M5_RUNS
M1=$M5_M1
B=$M5_B
FR=$M5_FR
FR_CLIPS="anime-clean anime-grain anime-dark cartoon-bright anime-sky anime-bright live-vfx live-slow"
mkdir -p "$OUT/numz" "$OUT/ours" "$OUT/logs"
TIMED="/usr/bin/time -v"
[ -x /usr/bin/time ] || { TIMED=""; echo "== no /usr/bin/time: no wall/RSS figures"; }

hold() {
    exec 9>>"$LOCK"
    local asked; asked=$(date +%s)
    flock 9
    echo "== lock held $(date -u +%FT%TZ) after $(( $(date +%s) - asked )) s; load $(cat /proc/loadavg)"
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader
    local check
    for check in 1 2 3; do
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
            | grep . || return 0
        sleep 5
    done
    echo "== another compute process shows: skipping"
    release
    return 1
}

release() {
    echo "== lock released $(date -u +%FT%TZ); load $(cat /proc/loadavg)"
    flock -u 9
    exec 9>&-
}

numz() {  # name input batch [bench.py options before --]
    local name=$1 input=$2 batch=$3; shift 3
    hold || { echo "STATUS numz $name skipped"; return; }
    local started; started=$(date +%s)
    (cd "$NUMZ" && .venv/bin/python "$SCRIPTS/bench.py" run "impl-m5-$name" --seedvr2-dir "$NUMZ" \
        --runs-dir "$RUNS" --results "$RUNS/impl-m5-results.jsonl" --overwrite \
        --env PYTHONDONTWRITEBYTECODE=1 \
        --env "FFV1_OUT_PATH=$OUT/numz/$name.mkv" --env FFV1_OUT_KEEP=0 \
        "$@" --wrap "$SCRIPTS/ffv1_out.py" -- \
        "$input" --output "$OUT/numz/$name.out/" --output_format png --video_backend ffmpeg \
        --model_dir "$MODELS" --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 \
        --seed 42 --attention_mode flash_attn_2 --batch_size "$batch" --temporal_overlap 0 \
        --color_correction lab --debug)
    local status=$?
    echo "STATUS numz $name exit $status in $(( $(date +%s) - started )) s"
    release
}

sampler() {  # the lab work dir's files, every 0.5 s: epoch size path
    while :; do
        find "$1" -type f -printf "$(date +%s.%N) %s %P\n" 2>/dev/null
        sleep 0.5
    done
}

ours() {  # name input cc [seedvr2x options]; inside a hold
    local name=$1 input=$2 cc=$3; shift 3
    local master="$OUT/ours/$name.mkv"
    echo "== ours $name ($cc) $(date -u +%FT%TZ); load $(cat /proc/loadavg)"
    sampler "$master.work" > "$OUT/logs/ours-$name.sizes" &
    local spid=$!
    (cd "$SX" && $TIMED .venv/bin/python -m seedvr2x "$input" -o "$master" \
        --model-dir "$MODELS" --dit-model seedvr2_ema_7b_fp16.safetensors --resolution 1080 \
        --seed 42 --color-correction "$cc" "$@") > "$OUT/logs/ours-$name.log" 2>&1
    local status=$?
    kill "$spid"; wait "$spid" 2>/dev/null
    grep -E "encoded in|window [0-9]+/[0-9]+, latents|decoded in|Elapsed|Maximum resident" \
        "$OUT/logs/ours-$name.log"
    echo "STATUS ours $name exit $status"
}

ours_alone() {  # name input cc [options]: one run in its own hold
    hold || { echo "STATUS ours $1 skipped"; return; }
    ours "$@"
    release
}

cost() {  # tag input: lab, none, lab, none back to back in ONE hold
    local tag=$1 input=$2
    hold || { echo "STATUS cost $tag skipped"; return; }
    ours "$tag-lab1" "$input" lab
    ours "$tag-none1" "$input" none
    ours "$tag-lab2" "$input" lab
    ours "$tag-none2" "$input" none
    release
}

numz_m1() {  # tests/test_lab.py's reference, by its docstring's command (gone with 68b1529)
    hold || { echo "STATUS numz m1 skipped"; return; }
    local started; started=$(date +%s)
    (cd "$NUMZ" && FFV1_OUT_PATH="$(dirname "$M1")/numz_lab.mkv" FFV1_OUT_KEEP=0 \
        .venv/bin/python "$SCRIPTS/ffv1_out.py" inference_cli.py "$M1" \
        --output "$OUT/numz/m1.out" --output_format png --model_dir "$MODELS" \
        --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 --batch_size 45 \
        --seed 42 --attention_mode flash_attn_2 --color_correction lab) \
        > "$OUT/logs/numz-m1.log" 2>&1
    local status=$?
    tail -n 5 "$OUT/logs/numz-m1.log"
    echo "STATUS numz m1 exit $status in $(( $(date +%s) - started )) s"
    release
}

case "${1:-}" in
m1)
    numz_m1
    cost m1 "$M1"
    ;;
rest)
    numz b "$B" 81
    cost b "$B"
    numz b-stitch "$B" 81 --env STITCH_LATENT=6:2 --env STITCH_CURVE=cosine \
        --wrap "$SCRIPTS/blend_patch.py"
    ours_alone b-w6 "$B" lab --window 6
    for clip in $FR_CLIPS; do
        numz "fr-$clip" "$FR/$clip.d1.lr.mkv" 45
        ours_alone "fr-$clip" "$FR/$clip.d1.lr.mkv" lab
    done
    ;;
*)
    usage
    ;;
esac
echo "== stage ${1} done $(date -u +%FT%TZ)"
