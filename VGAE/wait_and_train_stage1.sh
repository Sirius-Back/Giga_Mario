#!/usr/bin/env bash
set -eo pipefail
cd /home/User14
source /home/User14/miniconda3/etc/profile.d/conda.sh
conda activate caduceus_env
export PYTHONPATH=/home/User14

LOG=VGAE/stage1_region_k5/train.log
mkdir -p VGAE/stage1_region_k5

while true; do
  echo "=== $(date -Iseconds) polling for free GPU ===" | tee -a "$LOG"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
  DEV=$(python - <<'PY'
from src.pipeline.job_queue import can_launch_parallel, CLASS_GPU_TRAIN
import torch
# Prefer queue-launchable GPUs with lowest used memory
cands = []
for g in range(torch.cuda.device_count()):
    ok, reason = can_launch_parallel(peak_ram_gib=12.0, gpus=(g,), job_class=CLASS_GPU_TRAIN)
    if not ok:
        continue
    try:
        free, total = torch.cuda.mem_get_info(g)
        used = total - free
    except Exception:
        used = 10**18
    cands.append((used, g))
if not cands:
    print("")
else:
    cands.sort()
    print(f"cuda:{cands[0][1]}")
PY
)
  if [ -n "$DEV" ]; then
    echo "Launching Stage1 VGAE on $DEV" | tee -a "$LOG"
    exec python -u -m src.splits.vgae \
      --stage 1 \
      --out VGAE/stage1_region_k5 \
      --graph-dir runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph \
      --marked-dir ready_legnet/MARKED \
      --k 5 \
      --seed 42 \
      --min-epochs 25 \
      --patience 10 \
      --max-epochs 200 \
      --wait-poll-sec 600 \
      --device "$DEV" >>"$LOG" 2>&1
  fi
  echo "No free GPU; sleeping 600s" | tee -a "$LOG"
  sleep 600
done
