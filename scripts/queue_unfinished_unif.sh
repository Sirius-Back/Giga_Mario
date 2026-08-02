#!/usr/bin/env bash
# Queue unfinished runs_unif jobs that are not already RUNNING.
# Waits for free GPU + RAM<=90%; serializes after resubmit_waiters when present.
set -euo pipefail
cd /home/User14
LOG=logs/queue_unfinished_unif.log
mkdir -p logs
echo "queue_unfinished start $(date -Iseconds)" | tee -a "$LOG"

wait_gpu() {
  local thresh=${1:-1500}
  while true; do
    # Prefer not colliding with resubmit_waiters chain claiming GPUs
    if pgrep -f 'bash scripts/resubmit_waiters.sh' >/dev/null 2>&1; then
      echo "wait resubmit_waiters still alive $(date -Iseconds)" >>"$LOG"
      sleep 120
      continue
    fi
    used_pct=$(awk '/MemTotal:/ {t=$2} /MemAvailable:/ {a=$2} END {printf "%.0f", 100*(t-a)/t}' /proc/meminfo)
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    free_idx=""
    for i in 0 1 2 3; do
      u=${used[$i]:-99999}
      if [ "$u" -lt "$thresh" ]; then free_idx=$i; break; fi
    done
    if [ -n "$free_idx" ] && [ "$used_pct" -le 90 ]; then
      sleep 5
      mapfile -t used2 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      u2=${used2[$free_idx]:-99999}
      used_pct2=$(awk '/MemTotal:/ {t=$2} /MemAvailable:/ {a=$2} END {printf "%.0f", 100*(t-a)/t}' /proc/meminfo)
      if [ "$u2" -lt "$thresh" ] && [ "$used_pct2" -le 90 ]; then
        echo "$free_idx"
        return 0
      fi
    fi
    echo "wait GPUs used=${used[*]} RAM=${used_pct}% $(date -Iseconds)" >>"$LOG"
    sleep 60
  done
}

is_running() {
  local needle="$1"
  # Match real python module runs only (ignore bash script text / writers).
  ps -eo pid,cmd | awk -v n="$needle" '
    $0 ~ ("python -m src.runs_unif." n ".continue_from_split") {found=1}
    END {exit found?0:1}'
}

# --- run12: archive partial direct then retrain ---
if [ -f runs_unif/caduceus/run12_caduceus_kmer_k4/pipeline_done.json ]; then
  echo "run12 done — skip" | tee -a "$LOG"
elif is_running run12_caduceus_kmer_k4; then
  echo "run12 already RUNNING — skip" | tee -a "$LOG"
else
  d=runs_unif/caduceus/run12_caduceus_kmer_k4/direct
  if [ -d "$d" ] && [ ! -f "$d/best_model/best_meta.json" ]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    mv "$d" "runs_unif/caduceus/run12_caduceus_kmer_k4/direct_FAILED_PARTIAL_${stamp}"
    echo "archived partial run12 direct → direct_FAILED_PARTIAL_${stamp}" | tee -a "$LOG"
  fi
  GPU=$(wait_gpu 200)
  echo "run12 resubmit GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n caduceus_env --no-capture-output \
    python -m src.runs_unif.run12_caduceus_kmer_k4.continue_from_split \
    skip_wait=true \
    2>&1 | tee -a logs/run12_caduceus_resubmit.log | tee -a "$LOG" || echo "run12 FAIL" | tee -a "$LOG"
fi

# --- run33: LOO LegNet failed OOM; resubmit with internal wait ---
if [ -f runs_unif/legnet/run33_legnet_pangenome_k7_wm100_100_loo5/pipeline_done.json ]; then
  echo "run33 done — skip" | tee -a "$LOG"
elif is_running run33_legnet_pangenome_k7_wm100_100_loo5; then
  echo "run33 already RUNNING — skip" | tee -a "$LOG"
else
  GPU=$(wait_gpu 200)
  echo "run33 resubmit GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n legnet --no-capture-output \
    python -m src.runs_unif.run33_legnet_pangenome_k7_wm100_100_loo5.continue_from_split \
    skip_wait=true \
    2>&1 | tee -a logs/run33_legnet_resubmit.log | tee -a "$LOG" || echo "run33 FAIL" | tee -a "$LOG"
fi

# --- run14: restore archived split into outdir, then train ---
if [ -f runs_unif/caduceus/run14_caduceus_kmer_k7/pipeline_done.json ]; then
  echo "run14 done — skip" | tee -a "$LOG"
elif is_running run14_caduceus_kmer_k7; then
  echo "run14 already RUNNING — skip" | tee -a "$LOG"
else
  arch=runs_unif/caduceus/run14_caduceus_kmer_k7_ARCHIVED_MEGA_20260731T224550Z
  out=runs_unif/caduceus/run14_caduceus_kmer_k7
  if [ ! -f "$out/split_done.json" ] && [ -f "$arch/split_done.json" ]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    if [ -d "$out" ]; then
      mv "$out" "runs_unif/caduceus/run14_caduceus_kmer_k7_STUB_${stamp}"
    fi
    cp -a "$arch" "$out"
    echo "restored run14 split from archive → $out" | tee -a "$LOG"
  fi
  GPU=$(wait_gpu 200)
  echo "run14 resubmit GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n caduceus_env --no-capture-output \
    python -m src.runs_unif.run14_caduceus_kmer_k7.continue_from_split \
    skip_wait=true \
    2>&1 | tee -a logs/run14_caduceus_resubmit.log | tee -a "$LOG" || echo "run14 FAIL" | tee -a "$LOG"
fi

# run20/22 adv covered by scripts/resubmit_waiters.sh
# run32/34/36/39/41 already RUNNING

echo "queue_unfinished done $(date -Iseconds)" | tee -a "$LOG"
