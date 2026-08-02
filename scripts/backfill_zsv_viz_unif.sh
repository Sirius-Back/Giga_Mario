#!/usr/bin/env bash
# Backfill missing train-monitor figures and ZSV for finished / nearly-finished runs_unif.
set -euo pipefail
cd /home/User14
LOG=logs/backfill_zsv_viz_unif.log
mkdir -p logs
echo "backfill_zsv_viz start $(date -Iseconds)" | tee -a "$LOG"

has_monitor_figs() {
  local d="$1"
  compgen -G "$d/figures/train_monitor/Figure_*.pdf" >/dev/null 2>&1 \
    || compgen -G "$d/figures/train_monitor/manuscript/Figure_*.pdf" >/dev/null 2>&1 \
    || compgen -G "$d/figures/train_monitor/Figure_*.png" >/dev/null 2>&1
}

refresh_monitor() {
  local train_dir="$1"
  if has_monitor_figs "$train_dir"; then
    echo "viz OK — skip $train_dir" | tee -a "$LOG"
    return 0
  fi
  echo "viz refresh $train_dir $(date -Iseconds)" | tee -a "$LOG"
  conda run -n caduceus_env --no-capture-output \
    python -m src.train_viz.train_monitor --run-dir "$train_dir" \
    2>&1 | tee -a "$LOG" || echo "viz FAIL $train_dir" | tee -a "$LOG"
}

for td in \
  runs_unif/legnet/run24_legnet_paralogs_only/direct \
  runs_unif/legnet/run29_legnet_pangenome_k7_wm100_100/direct \
  runs_unif/legnet/run31_legnet_pangenome_k7_w0_100_loo5/adversarial/train \
  runs_unif/legnet/run35_legnet_loco/direct \
  runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/direct
do
  refresh_monitor "$td"
done

wait_gpu() {
  local thresh=${1:-1500}
  while true; do
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

for run in run36_caduceus_loco run39_caduceus_blastp; do
  root="runs_unif/caduceus/$run"
  out="$root/direct"
  zsv="$out/logs/zero_shot_metrics.json"
  if [ -f "$zsv" ]; then
    echo "zsv OK — skip $run" | tee -a "$LOG"
  elif [ -f "$out/best_model/best_meta.json" ] || [ -f "$out/best_model/config.json" ]; then
    GPU=$(wait_gpu 1500)
    echo "zsv eval $run GPU $GPU $(date -Iseconds)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n caduceus_env --no-capture-output \
      python -m src.pipeline.zsv_eval \
        --model caduceus --outdir "$out" --split-root "$root" --device 0 \
      2>&1 | tee -a "$LOG" || echo "zsv FAIL $run" | tee -a "$LOG"
  else
    echo "zsv skip $run — no best_model yet" | tee -a "$LOG"
  fi
  refresh_monitor "$out"
done

for run in \
  run1_caduceus_random \
  run25_caduceus_paralogs_only \
  run28_caduceus_pangenome_k7_w0_100 \
  run30_caduceus_pangenome_k7_wm100_100 \
  run38_caduceus_pangenome_k5_wm100_100
do
  z="runs_unif/caduceus/$run/adversarial/train/logs/zero_shot_metrics.json"
  if [ -f "$z" ]; then
    echo "adv zsv marker OK $run" | tee -a "$LOG"
  else
    echo "adv zsv MISSING $run — writing skip via zsv_eval" | tee -a "$LOG"
    conda run -n caduceus_env --no-capture-output \
      python -m src.pipeline.zsv_eval \
        --model caduceus \
        --outdir "runs_unif/caduceus/$run/adversarial/train" \
        --split-root "runs_unif/caduceus/$run" --device 0 \
      2>&1 | tee -a "$LOG" || true
  fi
done

echo "backfill_zsv_viz done $(date -Iseconds)" | tee -a "$LOG"
