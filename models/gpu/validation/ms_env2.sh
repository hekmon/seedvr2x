# Model conversation, slices A + B scoring (night of 2026-10-07): ms_env.sh's settings, plus slice B's (4K).
# Sourced by ms_jobs2.sh, ms_pool2.sh, ms_status2.sh (ms_env.sh itself unchanged: the old pool's scripts read it).
for v in VAL_GLUE MEAS_SHOTS; do [ -n "${!v:-}" ] || { echo "${0##*/}: $v is unset: source your glue.env (models/gpu/validation/glue.env.example)" >&2; exit 2; }; done
source "$VAL_GLUE/ms_env.sh"
C4=$MEAS_SHOTS                                 # the 4K shots (read only): <shot>.gt.mkv, <shot>.d1.bicubic.mkv
SHOTS_B="sollevante-painted sollevante-line sollevante-action sollevante-dark digital-sunrise digital-space"
# S9 (2026-10-07): slice B of the sharp 7B's files (labels sh-*): the 5 4K shots with two sharp fp16 seeds (colour's
# dumps/<shot>-d1/sh42 and sh43; GPU ids b-<shot>-d1-sh-<file>)
SHOTS_SH="digital-cockpit ouatia-face digital-space sollevante-painted cel4k-detail"
declare -A NFR=([sollevante-dark]=41)           # frames per 4K shot (else 45), as colour's NFR
# S9: the first film's 4K settings, colour's (b2_cpu_jobs.sh: --bars 42:42, bands rows 64:2096; the others whole)
BARS[ouatia-face]=42:42
ROWS[ouatia-face]=64:2096
T4=8   # threads per 4K job: colour's batch-A (b1_cpu_jobs3.sh TH=8) and baton-2 (b2_cpu_jobs.sh T4=8) 4K jobs'
