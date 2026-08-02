#!/usr/bin/env bash
set -eo pipefail
cd /home/User14
source /home/User14/miniconda3/etc/profile.d/conda.sh
conda activate caduceus_env
export PYTHONPATH=/home/User14
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
  --wait-poll-sec 600
