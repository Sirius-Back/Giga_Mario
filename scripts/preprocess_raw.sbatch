#!/bin/bash
#SBATCH --job-name=preprocess_raw
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/preprocess_raw_%j.out
#SBATCH --error=logs/preprocess_raw_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs data_ready

conda run -n caduceus_env python src/preprocessing.py \
  --raw raw \
  --out data_ready \
  --flank 10000 \
  --seed 42
