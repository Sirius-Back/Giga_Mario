#!/usr/bin/env bash
# Wait for one free GPU, then: missing ZSV evals + LegNet best repredict + compare plots.
# Wait logs go to stderr/file only so GPU id capture stays clean.
set -euo pipefail
cd /home/User14
LOG=logs/best_split_metrics_zsv_waiter2.log
echo "waiter2 start $(date -Iseconds)" | tee -a "$LOG"

wait_gpu() {
  while true; do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    free_idx=""
    for i in 0 1 2 3; do
      u=${used[$i]:-99999}
      if [ "$u" -lt 1500 ]; then free_idx=$i; break; fi
    done
    if [ -n "$free_idx" ]; then
      sleep 5
      mapfile -t used2 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      u2=${used2[$free_idx]:-99999}
      if [ "$u2" -lt 1500 ]; then
        echo "$free_idx"
        return 0
      fi
    fi
    echo "wait GPUs used=${used[*]} $(date -Iseconds)" >>"$LOG"
    sleep 60
  done
}

GPU=$(wait_gpu)
echo "GPU $GPU free $(date -Iseconds)" | tee -a "$LOG"

# LegNet best-ckpt repredict (train/val/test Spearman)
echo "LegNet best_split_metrics on GPU $GPU" | tee -a "$LOG"
conda run -n caduceus_env --no-capture-output python -m src.pipeline.best_split_metrics \
  --legnet-only --force --device "$GPU" \
  2>&1 | tee -a "$LOG" || true

conda run -n caduceus_env --no-capture-output python -m src.pipeline.best_split_metrics --caduceus-only --force 2>&1 | tee -a "$LOG" || true
conda run -n caduceus_env --no-capture-output python -m src.pipeline.compare_best_models 2>&1 | tee -a "$LOG" || true
echo "waiter2 done $(date -Iseconds)" | tee -a "$LOG"
