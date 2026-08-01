#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
LOG=logs/best_split_metrics_zsv_waiter.log
echo "expanded waiter start $(date -Iseconds)" | tee -a "$LOG"

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
    echo "wait GPUs used=${used[*]} $(date -Iseconds)" | tee -a "$LOG"
    sleep 60
  done
}

GPU=$(wait_gpu)
echo "GPU $GPU free $(date -Iseconds)" | tee -a "$LOG"

# 1) Missing Caduceus ZSV on trained best models (regression only)
for item in \
  "caduceus|runs_unif/caduceus/run20_caduceus_pangenome_k10_w0_100/direct|runs_unif/caduceus/run20_caduceus_pangenome_k10_w0_100" \
  "caduceus|runs_unif/caduceus/run22_caduceus_pangenome_k10_wm100_100/direct|runs_unif/caduceus/run22_caduceus_pangenome_k10_wm100_100" \
  "caduceus|runs/run18_pangenome_CDS_caduceus/direct|runs/run18_pangenome_CDS_caduceus"
do
  IFS='|' read -r model outdir splitroot <<<"$item"
  zsv="$outdir/logs/zero_shot_metrics.json"
  if [ -f "$zsv" ] && ! grep -q '"skipped": true' "$zsv" 2>/dev/null && grep -q spearman "$zsv" 2>/dev/null; then
    echo "skip ZSV (present) $outdir" | tee -a "$LOG"
    continue
  fi
  # ensure best_model usable as final if final empty
  if [ ! -f "$outdir/final_model/config.json" ] && [ -f "$outdir/best_model/config.json" ]; then
    mkdir -p "$outdir/final_model"
    # prefer symlink-like copy of weights without destroying best
    rsync -a --delete "$outdir/best_model/" "$outdir/final_model/" || cp -a "$outdir/best_model/." "$outdir/final_model/"
    echo "staged best→final for ZSV $outdir" | tee -a "$LOG"
  fi
  echo "ZSV eval $outdir on GPU $GPU" | tee -a "$LOG"
  conda run -n caduceus_env --no-capture-output python -m src.pipeline.zsv_eval \
    --model "$model" --outdir "$outdir" --split-root "$splitroot" --device "$GPU" \
    2>&1 | tee -a "$LOG" || echo "ZSV FAIL $outdir" | tee -a "$LOG"
done

# 2) LegNet best-ckpt repredict (train/val/test)
echo "LegNet best_split_metrics on GPU $GPU" | tee -a "$LOG"
conda run -n caduceus_env --no-capture-output python -m src.pipeline.best_split_metrics \
  --legnet-only --force --device "$GPU" \
  runs_unif/legnet/*/direct \
  runs_unif/legnet/*/adversarial/train \
  runs/run5/direct \
  runs/run5/adversarial/train \
  2>&1 | tee -a "$LOG" || true

# 3) Refresh caduceus extracts + comparison plots
conda run -n caduceus_env --no-capture-output python -m src.pipeline.best_split_metrics --caduceus-only --force 2>&1 | tee -a "$LOG" || true
conda run -n caduceus_env --no-capture-output python -m src.pipeline.compare_best_models 2>&1 | tee -a "$LOG" || true
echo "waiter done $(date -Iseconds)" | tee -a "$LOG"
