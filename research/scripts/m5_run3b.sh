#!/bin/bash
# Milestone 5's step 3b: ours' side again (seedvr2x/DESIGN.md, Validation milestones 5), lab's
# reference now built in float32 (seedvr2x/DESIGN.md, Colour correction, Numerics), on every
# material: milestone 1's input, clip B, clip B in 6:2 windows, the 8 full-reference clips. Same
# options as m5_run.sh's ours runs (one batch, seed 42, 1080p, the 7B fp16), the master format
# named. Each run in its own hold of the GPU lock, the nvidia-smi check inside (m5_run.sh's hold
# and release). m5_score3b.sh scores the masters (ours3b/<name>.mkv).
#
# Historical: the record of what ran on 2026-10-04 (06:04-08:39Z), kept to re-run it: seedvr2x at
# eca0ff1, the commit that made lab's reference float32; 11 of 11 materials passed. seedvr2x's
# `--color-correction lab` went with 68b1529, when split replaced lab. numz's side is
# m5_run.sh's (numz 4490bd1 through research/scripts' bench.py and ffv1_out.py); this one calls
# no research script, seedvr2x's CLI alone. Clip B's input: m5_run.sh's header.
#
# The first line records M5_SEEDVR2X's tree, through a scratch index in M5_OUT/logs (the copy run
# then was a git work tree of seedvr2x/ alone).
set -u

usage() {
    cat >&2 <<'EOF'
usage: m5_run3b.sh
Environment, every variable required, paths absolute (the runs cd into the checkout):
  M5_OUT       where the runs go: ours3b/, logs/
  M5_SEEDVR2X  seedvr2x's project directory (its .venv), at eca0ff1
  M5_MODELS    the model directory: seedvr2_ema_7b_fp16.safetensors and numz's VAE
  M5_LOCK      the GPU lock file every run holds
  M5_M1        milestone 1's input_rgb.mkv
  M5_B         clip B's input_b.mkv
  M5_FR        the full-reference clips' directory (<clip>.d1.lr.mkv)
ffmpeg and ffprobe on PATH: the runs' were n9.0.2 with zimg.
EOF
    exit 2
}
[ $# = 0 ] || usage
for v in M5_OUT M5_SEEDVR2X M5_MODELS M5_LOCK M5_M1 M5_B M5_FR; do
    [ -n "${!v:-}" ] || { echo "m5_run3b.sh: $v is not set" >&2; usage; }
    [[ ${!v} == /* ]] || { echo "m5_run3b.sh: $v is not an absolute path: ${!v}" >&2; usage; }
done

LOCK=$M5_LOCK
OUT=$M5_OUT
SX=$M5_SEEDVR2X
MODELS=$M5_MODELS
M1=$M5_M1
B=$M5_B
FR=$M5_FR
FR_CLIPS="anime-clean anime-grain anime-dark cartoon-bright anime-sky anime-bright live-vfx live-slow"
mkdir -p "$OUT/ours3b" "$OUT/logs"
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

ours() {  # name input [seedvr2x options]: one run in its own hold
    local name=$1 input=$2; shift 2
    local master="$OUT/ours3b/$name.mkv"
    hold || { echo "STATUS ours3b $name skipped"; return; }
    echo "== ours3b $name $(date -u +%FT%TZ); load $(cat /proc/loadavg)"
    (cd "$SX" && $TIMED .venv/bin/python -m seedvr2x "$input" -o "$master" \
        --model-dir "$MODELS" --dit-model seedvr2_ema_7b_fp16.safetensors --resolution 1080 \
        --seed 42 --color-correction lab --format gbrp16le "$@") > "$OUT/logs/ours3b-$name.log" 2>&1
    local status=$?
    release
    grep -E "encoded in|window [0-9]+/[0-9]+, latents|decoded in|Elapsed|Maximum resident" \
        "$OUT/logs/ours3b-$name.log"
    echo "STATUS ours3b $name exit $status"
}

echo "== run3b start $(date -u +%FT%TZ); seedvr2x tree $(cd "$SX" && GIT_INDEX_FILE="$OUT/logs/tree.idx" git add -A . && GIT_INDEX_FILE="$OUT/logs/tree.idx" git write-tree)"
ours m1 "$M1"
ours b "$B"
ours b-w6 "$B" --window 6
for clip in $FR_CLIPS; do
    ours "fr-$clip" "$FR/$clip.d1.lr.mkv"
done
echo "== run3b done $(date -u +%FT%TZ)"
