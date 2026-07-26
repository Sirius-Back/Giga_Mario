#!/bin/bash
#SBATCH --job-name=split_random
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/split_random_%j.out
#SBATCH --error=logs/split_random_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

python -u -m src.splits.main \
  --strategy random \
  --raw raw \
  --ready ready \
  --seed 42 \
  --out splits/random
