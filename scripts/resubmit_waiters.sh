#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
LOG=logs/resubmit_waiters.log
echo "resubmit_waiters start $(date -Iseconds)" | tee -a "$LOG"

wait_gpu() {
  local thresh=${1:-1500}
  while true; do
    # also require RAM used <= 90% before claiming GPU
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

# --- 1) LegNet best-ckpt repredict (once) ---
if ! python3 - <<'PY'
import json
from pathlib import Path
p=Path('runs_unif/legnet/run2_legnet_random/direct/best_split_metrics.json')
d=json.loads(p.read_text()) if p.is_file() else {}
raise SystemExit(0 if d.get('source')=='best_ckpt_repredict' and d.get('spearman',{}).get('train') is not None else 1)
PY
then
  GPU=$(wait_gpu 1500)
  echo "LegNet best_split on GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  conda run -n legnet --no-capture-output python -m src.pipeline.best_split_metrics \
    --legnet-only --force --device "$GPU" 2>&1 | tee -a "$LOG" || true
  conda run -n caduceus_env --no-capture-output python -m src.pipeline.best_split_metrics --caduceus-only --force 2>&1 | tee -a "$LOG" || true
  conda run -n caduceus_env --no-capture-output python -m src.pipeline.compare_best_models 2>&1 | tee -a "$LOG" || true
else
  echo "LegNet best_split already complete — skip" | tee -a "$LOG"
fi

# --- 2) run20 adv resubmit ---
if [ ! -f runs_unif/caduceus/run20_caduceus_pangenome_k10_w0_100/adversarial/train/best_model/best_meta.json ]; then
  GPU=$(wait_gpu 1500)
  echo "run20 adv resubmit GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n caduceus_env --no-capture-output \
    python -m src.runs_unif.run20_caduceus_pangenome_k10_w0_100.continue_from_split \
    skip_direct=true force_adv=true skip_wait=true \
    2>&1 | tee -a logs/run20_caduceus_adv_resubmit.log | tee -a "$LOG" || echo "run20 FAIL" | tee -a "$LOG"
else
  echo "run20 adv already has best — skip" | tee -a "$LOG"
fi

# --- 3) run22 adv resubmit ---
if [ ! -f runs_unif/caduceus/run22_caduceus_pangenome_k10_wm100_100/adversarial/train/best_model/best_meta.json ]; then
  GPU=$(wait_gpu 1500)
  echo "run22 adv resubmit GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n caduceus_env --no-capture-output \
    python -m src.runs_unif.run22_caduceus_pangenome_k10_wm100_100.continue_from_split \
    skip_direct=true force_adv=true skip_wait=true \
    2>&1 | tee -a logs/run22_caduceus_adv_resubmit.log | tee -a "$LOG" || echo "run22 FAIL" | tee -a "$LOG"
else
  echo "run22 adv already has best — skip" | tee -a "$LOG"
fi

# --- 4) run38 with raised max_fold_size (source run37 has large modularity folds) ---
if pgrep -f 'src.runs_unif.run38_caduceus_pangenome_k5_wm100_100.continue_from_split' >/dev/null; then
  echo "run38 already RUNNING — skip" | tee -a "$LOG"
elif [ ! -f runs_unif/caduceus/run38_caduceus_pangenome_k5_wm100_100/pipeline_done.json ] \
   && [ ! -f runs_unif/caduceus/run38_caduceus_pangenome_k5_wm100_100/split_done.json ]; then
  GPU=$(wait_gpu 1500)
  echo "run38 resubmit max_fold_size=7000 GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU conda run -n caduceus_env --no-capture-output \
    python -m src.runs_unif.run38_caduceus_pangenome_k5_wm100_100.continue_from_split \
    max_fold_size=7000 skip_wait=true \
    2>&1 | tee -a logs/run38_caduceus_resubmit.log | tee -a "$LOG" || echo "run38 FAIL" | tee -a "$LOG"
else
  echo "run38 already staged/done — skip" | tee -a "$LOG"
fi

echo "resubmit_waiters done $(date -Iseconds)" | tee -a "$LOG"
