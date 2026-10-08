#!/bin/bash
# Model conversation, slices A + B scoring: the pool's status (ms_pool2.sh's; ms_status.sh's, with slice B).
#   bash $VAL_GLUE/ms_status2.sh [-q]      -q: counts, running jobs and failures only
for v in VAL_GLUE VAL_STATE VAL_DISK; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
source "$VAL_GLUE/ms_env2.sh"
q=${1:-}
now=$(date +%s)
pool=$(pgrep -f "bash $VAL_GLUE/ms_pool2?.sh" | tr '\n' ' ')
echo "== $(date -u +%FT%TZ) (Paris $(TZ=Europe/Paris date +%H:%M)); pool: ${pool:-not running} ($(for p in $pool; do tr '\0' ' ' < /proc/$p/cmdline | awk '{print $2}' | sed 's|.*/||'; done | tr '\n' ' '))$([ -e "$SC/STOP" ] && echo ', STOP set')"
bash "$VAL_GLUE/ms_jobs2.sh" 2>/dev/null > "$SC/status-jobs.tsv"
pat='sollevante-|digital-sunrise|digital-space|digital-cockpit|ouatia-face|cel4k-detail'  # S9: the sharp's B shots too
echo "-- jobs by state, 1080p: $(grep -Pv "^[^\t]*($pat)" "$SC/status-jobs.tsv" | cut -f3 | sort | uniq -c | awk '{printf "%s %s, ", $2, $1}')"
echo "-- jobs by state, 4K:    $(grep -P "^[^\t]*($pat)" "$SC/status-jobs.tsv" | cut -f3 | sort | uniq -c | awk '{printf "%s %s, ", $2, $1}')"
echo "-- GPU markers: a- $(ls "$GPUDONE" 2>/dev/null | grep -c '^a-') done, $(ls "$GPUFAIL" 2>/dev/null | grep -c '^a-') failed; b- $(ls "$GPUDONE" 2>/dev/null | grep -c '^b-') done, $(ls "$GPUFAIL" 2>/dev/null | grep -c '^b-') failed"
echo "-- running:"
for f in "$SC"/running/*; do
  [ -e "$f" ] || continue
  read -r pid kind s0 < "$f"
  echo "   ${f##*/} ($kind) $(( (now - s0) / 60 )) min, peak $(cat "$SC/peak/${f##*/}" 2>/dev/null || echo ?) MB, pid $pid $(kill -0 "$pid" 2>/dev/null && echo alive || echo GONE)"
done
nf=$(ls "$SC/failed" | wc -l)
if (( nf )); then echo "-- FAILED ($nf):"; for f in "$SC"/failed/*; do echo "   ${f##*/}: $(grep -v '^\$' "$SC/logs/${f##*/}.log" 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-200)"; done; fi
echo "-- times by kind (s: n mean max; peak MB max):"
awk -F'\t' '$7=="done" {n[$2]++; s[$2]+=$5; if ($5>m[$2]) m[$2]=$5; if ($6>p[$2]) p[$2]=$6} END {for (k in n) printf "   %s: %d, %.0f, %d; %d\n", k, n[k], s[k]/n[k], m[k], p[k]}' "$SC/times.tsv" 2>/dev/null | sort
echo "-- memory: $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) GB available; load $(cut -d' ' -f1-3 /proc/loadavg); $VAL_DISK $(df -h --output=avail "$VAL_DISK" | tail -1)"
echo "-- summaries: $(ls -la --time-style=+%H:%M $VAL_STATE/sum/*.md 2>/dev/null | awk '{print $6, $7}' | sed "s|$VAL_STATE/sum/||" | tr '\n' ' ')"
echo "-- pool log tail:"; tail -4 $VAL_STATE/95-pool.log 2>/dev/null
if [ "$q" != -q ]; then echo "-- waiting/ready:"; awk -F'\t' '$3=="wait" || $3=="ready" || $3=="blocked" {print "   " $1 " " $3 " " $4}' "$SC/status-jobs.tsv" | head -60; fi
