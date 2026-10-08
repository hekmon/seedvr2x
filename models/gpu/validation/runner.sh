#!/bin/bash
# The model conversation's GPU queue runner (night of 2026-10-07), after colour's runner_b2.sh.
#   bash $VAL_GLUE/runner.sh 2>&1 | tee -a $VAL_STATE/85-runner.log
# Jobs: $G/jobs.tsv (G=$VAL_STATE/gpu), one "id<TAB>command" per line ('#' comments and blank lines
# skipped; an id is letters, digits, '.', '_', '-'), read again before every job: append later slices with >>
# while the runner runs (never rewrite the file in place; an unterminated last line is not read yet).
# A job is pending while neither $G/done/<id> nor $G/failed/<id> exists; pending jobs run in file order.
# To run a failed job again: rm $G/failed/<id> (the runner takes it at its next turn).
# One job per GPU lock hold: mkdir -p ${VAL_GPU_LOCK%/*} && flock $VAL_GPU_LOCK bash -c HOLD. Inside the hold,
# before the job: nvidia-smi must list no compute process (else the runner stops, status 9: something runs on
# the GPU outside the lock); $VAL_DISK must have at least MIN_FREE_GB (60) GB free (else the runner stops,
# status 8); one line of the GPU's state (SM clock, clock-event reasons, power, temperature, memory) goes to
# $G/gpu-runstart.csv and to the head of the job's log. The job runs as `bash -c COMMAND` in $NUMZ_DIR,
# stdin /dev/null, its output appended to $G/logs/<id>.log; then $G/done/<id> or $G/failed/<id> (holding the
# exit status, start, end, seconds; written whole, by rename) and one line in $G/gpu-jobs.csv. A failed job
# never stops the runner.
# Between jobs: $G/STOP stops the runner (status 3; a running job is never touched). No job starts at or after
# CUTOFF (08:45 Paris = 2026-10-07T06:45:00Z) unless $G/GO_PAST_0845 exists: the runner then waits, polling
# every 60 s, and goes on as soon as the file appears (or stops on STOP). With no job pending it polls the job
# file every 60 s: it never exits on an empty queue. $G/RUNNING names the job in the hold (for status.sh).
# One runner at a time ($G/runner.lock). Status: bash $VAL_GLUE/status.sh
set -u
for v in VAL_STATE VAL_GPU_LOCK VAL_DISK NUMZ_DIR; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
G=$VAL_STATE/gpu
JOBS=$G/jobs.tsv
LOCK=$VAL_GPU_LOCK
STOP=$G/STOP
GO=$G/GO_PAST_0845
CUTOFF=${CUTOFF:-2026-10-07T06:45:00Z}
MIN_FREE_GB=${MIN_FREE_GB:-60}
POLL=60
ts() { date -u +%FT%TZ; }
cut=$(date -u -d "$CUTOFF" +%s) || { echo "runner: bad CUTOFF $CUTOFF"; exit 2; }
mkdir -p "$G/done" "$G/failed" "$G/logs" "${LOCK%/*}"
exec 8> "$G/runner.lock"
flock -n 8 || { echo "$(ts) runner: another runner holds $G/runner.lock: exiting"; exit 4; }
[ -s "$G/gpu-runstart.csv" ] || echo "utc,id,timestamp,clocks.sm,clocks_event_reasons.active,power.draw,temperature.gpu,memory.used" > "$G/gpu-runstart.csv"
[ -s "$G/gpu-jobs.csv" ] || echo "id,exit_status,start_utc,end_utc,seconds" > "$G/gpu-jobs.csv"
export G MIN_FREE_GB VAL_DISK NUMZ_DIR
HOLD='set -u
id=$1 cmd=$2 log=$G/logs/$1.log
ts() { date -u +%FT%TZ; }
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
if [ -n "$apps" ]; then echo "$(ts) ABORT before $id: a compute process runs outside the lock: $apps"; exit 9; fi
free=$(df -BG --output=avail "$VAL_DISK" | tail -n 1 | tr -dc 0-9)
if [ -z "$free" ] || (( free < MIN_FREE_GB )); then echo "$(ts) STOP before $id: ${free:-?} GB free on $VAL_DISK (< $MIN_FREE_GB)"; exit 8; fi
q="--query-gpu=timestamp,clocks.sm,clocks_event_reasons.active,power.draw,temperature.gpu,memory.used"
smi=$(nvidia-smi $q --format=csv,noheader 2>&1 | head -n 1)
t0=$(ts) s0=$(date +%s)
printf "%s\t%s\t%s\t%s\n" "$id" "$t0" "$s0" "$$" > "$G/RUNNING"
echo "$t0,$id,${smi//, /,}" >> "$G/gpu-runstart.csv"
{ echo "=== $t0 runner: start $id (${free} GB free on $VAL_DISK)"; echo "=== GPU at start: $smi"; echo "=== command: $cmd"; } >> "$log"
echo "$t0 start $id; GPU: $smi"
( cd "$NUMZ_DIR" && bash -c "$cmd" ) >> "$log" 2>&1 < /dev/null
rc=$?
t1=$(ts) s1=$(date +%s) d=$(( $(date +%s) - s0 ))
smi1=$(nvidia-smi $q --format=csv,noheader 2>&1 | head -n 1)
echo "=== $t1 runner: end $id, exit $rc, $d s; GPU at end: $smi1" >> "$log"
echo "$id,$rc,$t0,$t1,$d" >> "$G/gpu-jobs.csv"
m=$G/.marker.$$
printf "exit %s\nstart %s\nend %s\nseconds %s\n" "$rc" "$t0" "$t1" "$d" > "$m"
if (( rc == 0 )); then mv -f "$m" "$G/done/$id"; echo "$t1 done $id ($d s)"
else mv -f "$m" "$G/failed/$id"; echo "$t1 FAILED $id: exit $rc ($d s), log $log"; fi
rm -f "$G/RUNNING"
exit 0'
echo "$(ts) runner pid $$ on $JOBS: one job per hold of $LOCK; markers $G/done, $G/failed; stop file $STOP; no job starts from $CUTOFF without $GO; at least $MIN_FREE_GB GB free on $VAL_DISK"
idle="" late=""
declare -A bad=()
while :; do
  if [ -e "$STOP" ]; then echo "$(ts) $STOP exists: the runner stops between jobs"; exit 3; fi
  next="" left=0
  if [ -f "$JOBS" ]; then
    while IFS=$'\t' read -r id cmd; do
      [[ -z "${id// /}" || "$id" == \#* ]] && continue
      if [[ ! "$id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || -z "${cmd// /}" ]]; then
        [ -n "${bad[$id]:-}" ] || echo "$(ts) skipping a malformed line (id '$id')"
        bad[$id]=1; continue
      fi
      [[ -e "$G/done/$id" || -e "$G/failed/$id" ]] && continue
      left=$(( left + 1 ))
      [ -n "$next" ] || next="$id"$'\t'"$cmd"
    done < "$JOBS"
  fi
  if [ -z "$next" ]; then
    [ -n "$idle" ] || echo "$(ts) no job pending: polling $JOBS every $POLL s"
    idle=1 late=""; sleep $POLL; continue
  fi
  idle=""
  if (( $(date +%s) >= cut )) && [ ! -e "$GO" ]; then
    [ -n "$late" ] || echo "$(ts) past $CUTOFF (08:45 Paris): no job starts until $GO exists ($left pending); polling every $POLL s"
    late=1; sleep $POLL; continue
  fi
  [ -z "$late" ] || echo "$(ts) $GO exists: going on"
  late=""
  IFS=$'\t' read -r id cmd <<< "$next"
  echo "$(ts) next $id ($left pending): waiting for the GPU lock"
  mkdir -p "${LOCK%/*}" && flock "$LOCK" bash -c "$HOLD" hold "$id" "$cmd"
  st=$?
  if (( st != 0 )); then echo "$(ts) the runner stops (status $st)"; exit $st; fi
  if [[ ! -e "$G/done/$id" && ! -e "$G/failed/$id" ]]; then echo "$(ts) $id left no marker: the runner stops"; exit 5; fi
done
