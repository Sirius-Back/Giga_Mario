#!/usr/bin/env bash
# Kill continue_from_split only when log tail indicates adversarial phase (not direct/fold train).
set -uo pipefail
cd /home/User14
LOG=logs/kill_adversarial_watcher.log
mkdir -p logs

RUNS=(
  run32_caduceus_pangenome_k7_w0_100_loo5
  run34_caduceus_pangenome_k7_wm100_100_loo5
  run42_caduceus_vgae_stage1_k5
  run39_caduceus_blastp
  run20_caduceus_pangenome_k10_w0_100
  run22_caduceus_pangenome_k10_wm100_100
)

log() { echo "$1" | tee -a "$LOG"; }

in_adversarial_phase() {
  local logfile="$1"
  [ -f "$logfile" ] || return 1
  tail -n 80 "$logfile" 2>/dev/null | grep -qE 'adversarial: copy|adversarial Caduceus train'
}

archive_adv() {
  local out="$1"
  local adv="$out/adversarial"
  [ -d "$adv" ] || return 0
  if [ -f "$adv/best_meta.json" ] || [ -f "$adv/train/best_model/best_meta.json" ]; then
    return 0
  fi
  local stamp
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  mv "$adv" "${out}/adversarial_SKIPPED_${stamp}"
  log "archived ${adv} -> adversarial_SKIPPED_${stamp}"
}

write_pipeline_done_if_missing() {
  local run="$1"
  local out="$2"
  local done="$out/pipeline_done.json"
  [ -f "$done" ] && return 0
  python3 - "$run" "$out" <<'PY'
import json, sys
from datetime import datetime, timezone
run, out = sys.argv[1], sys.argv[2]
p = __import__("pathlib").Path(out) / "pipeline_done.json"
p.write_text(json.dumps({
    "status": "COMPLETED",
    "run_name": run,
    "out_root": out,
    "adversarial": False,
    "note": "adversarial skipped by user request (watcher)",
    "finished_at": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
print(p)
PY
  log "wrote pipeline_done for $run"
}

log "kill_adversarial_watcher start $(date -Iseconds)"
while true; do
  for run in "${RUNS[@]}"; do
    needle="python -m src.runs_unif.${run}.continue_from_split"
    if ! pgrep -f "$needle" >/dev/null 2>&1; then
      continue
    fi
    # Resolve log path (run42 uses non-_pipeline name)
    logfile="logs/${run}_pipeline.log"
    if [ ! -f "$logfile" ]; then
      logfile="logs/${run}.log"
    fi
    if ! in_adversarial_phase "$logfile"; then
      continue
    fi
    log "ADV phase detected for $run — killing continue_from_split (log=$logfile)"
    pkill -f "src.runs_unif.${run}.continue_from_split" || true
    sleep 2
    pkill -9 -f "src.runs_unif.${run}.continue_from_split" || true
    out="runs_unif/caduceus/${run}"
    [ -d "$out" ] || out="runs_unif/legnet/${run}"
    archive_adv "$out"
    write_pipeline_done_if_missing "$run" "$out"
  done
  sleep 60
done
