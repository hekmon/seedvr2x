#!/bin/bash
# Model conversation, slices A (1080p) and B (4K) scoring: the CPU pool (ms_pool.sh's, with slice B's 4K jobs).
# Run in the foreground of a terminal of its own:
#   bash $VAL_GLUE/ms_pool2.sh 2>&1 | tee -a $VAL_STATE/95-pool.log
# Every 20 s: the job list (ms_jobs2.sh: every GPU run marker a-* and b-* in $VAL_STATE/gpu/done/, the extra
# labels, the 7B's references), then ready jobs started in list order, each in its own session (ms_job.sh, unchanged:
# cores 0-31, nice 19), within NMAX slots (file $SC/NMAX overrides, default 6; a 1080p job = 1 slot (4 torch threads),
# a 4K job = 2 slots (8 threads)), and only while MemAvailable stays above MINGB (60) after the running jobs'
# expected peaks; strict priority (the first ready job that does not fit stops the turn). After any job ended, the
# summaries re-made (ms_sum.py for slice A, then ms_sum4k.py for slice B, then ms_sum4ksh.py for slice B of the
# sharp's files, S9; timeout 15 min each).
# Stop: touch $VAL_STATE/score/STOP -> starts nothing more, waits for the running jobs, re-makes the
# summaries, exits (remove STOP before a restart). A restart adopts running jobs (running/<id> with a live pid);
# a job whose pid died without a marker (box reboot) is marked failed (rm failed/<id> to run it again).
# Ctrl-C here only stops the pool (no trap): the jobs run in their own sessions and finish on their own.
set -u
for v in VAL_GLUE VAL_STATE; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
source "$VAL_GLUE/ms_env2.sh"
MINGB=${MINGB:-60}
# GB expected (colour's 1080p: sc 14, vm 8, bd 3; fr 13.2; colour's 4K two-variant jobs at 8 threads, 2026-10-06:
# sc4 44-54, vm4 27-32, bd4 7-11)
declare -A PEAK=([sc]=16 [vm]=12 [bd]=4 [df]=2 [fr]=15 [sc4]=58 [vm4]=34 [bd4]=13 [fr4]=60)
declare -A W=([sc4]=2 [vm4]=2 [bd4]=2 [fr4]=2)   # slots (else 1)
mkdir -p "$SC"/{done,failed,running,peak,logs} "$G"/{eval,vmaf,masters,bands,diff,fr,tmp} $VAL_STATE/sum
say() { echo "$(date -u +%FT%TZ) pool: $*"; }
summarize() {
  local s
  for s in ms_sum.py ms_sum4k.py ms_sum4ksh.py; do  # S9: ms_sum4ksh.py = slice B of the sharp's files
    [ -f "$VAL_GLUE/$s" ] || continue
    say "summaries ($s): start"
    if timeout 900 taskset -c 0-31 nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
         $PY "$VAL_GLUE/$s" > "$SC/logs/${s%.py}.log" 2>&1; then
      say "summaries ($s): done ($(tail -1 "$SC/logs/${s%.py}.log"))"
    else
      say "summaries ($s): FAILED (exit $?), see $SC/logs/${s%.py}.log: $(tail -2 "$SC/logs/${s%.py}.log" | tr '\n' ' ')"
    fi
  done
}
nrunning() {  # prunes stale running files, prints the slots taken and the reserved GB
  local f pid kind s0 n=0 gb=0 id
  for f in "$SC"/running/*; do
    [ -e "$f" ] || continue
    read -r pid kind s0 < "$f" || continue
    id=${f##*/}
    if ! kill -0 "$pid" 2>/dev/null; then
      sleep 1
      [ -e "$f" ] || continue  # it just ended
      if [ ! -e "$SC/done/$id" ] && [ ! -e "$SC/failed/$id" ]; then
        say "stale $id (pid $pid gone, no marker): marked failed" >&2; touch "$SC/failed/$id"
      fi
      rm -f "$f"; continue
    fi
    n=$(( n + ${W[$kind]:-1} )); gb=$(( gb + ${PEAK[$kind]:-16} ))
  done
  echo "$n $gb"
}
say "ms_pool2 start (pid $$), NMAX ${NMAX:-6} slots (file $SC/NMAX overrides; a 4K job 2), MINGB $MINGB, FR_ON $([ -e "$SC/FR_ON" ] && echo yes || echo no), FR4_ON $([ -e "$SC/FR4_ON" ] && echo yes || echo no); adopting: $(ls "$SC/running" | tr '\n' ' ')"
last=-1
while :; do
  ndone=$(ls "$SC/done" | wc -l) nfail=$(ls "$SC/failed" | wc -l)
  if (( ndone + nfail != last )); then
    (( last >= 0 )) && summarize
    last=$(( ndone + nfail ))
  fi
  read -r nrun gbres < <(nrunning)
  if [ -e "$SC/STOP" ]; then
    if (( nrun == 0 )); then
      ndone=$(ls "$SC/done" | wc -l) nfail=$(ls "$SC/failed" | wc -l)
      (( ndone + nfail != last )) && summarize
      say "STOP: no job running, exits"; exit 0
    fi
    sleep 20; continue
  fi
  nmax=$(cat "$SC/NMAX" 2>/dev/null || echo "${NMAX:-6}")
  if bash "$VAL_GLUE/ms_jobs2.sh" > "$SC/jobs.tsv.tmp" 2> "$SC/jobs.err"; then
    mv -f "$SC/jobs.tsv.tmp" "$SC/jobs.tsv"
  else
    say "ms_jobs2.sh failed: $(tail -2 "$SC/jobs.err" | tr '\n' ' ')"; sleep 20; continue
  fi
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  while IFS=$'\t' read -r id kind state why cmd; do
    [ "$state" = ready ] || continue
    w=${W[$kind]:-1}
    (( nrun + w <= nmax )) || break
    need=${PEAK[$kind]:-16}
    if (( avail - gbres - need < MINGB )); then say "memory: $avail GB available, $gbres reserved, $id needs $need: waits"; break; fi
    setsid bash "$VAL_GLUE/ms_job.sh" "$id" "$kind" "$cmd" < /dev/null > /dev/null 2>&1 &
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -e "$SC/running/$id" ] && break; sleep 0.2; done
    nrun=$(( nrun + w )) gbres=$(( gbres + need ))
    say "start $id ($kind), $nrun/$nmax slots"
  done < "$SC/jobs.tsv"
  sleep 20
done
