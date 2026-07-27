#!/bin/bash
#SBATCH --job-name=caduceus_full_random
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/caduceus_full_random_%j.out
#SBATCH --error=logs/caduceus_full_random_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/home/User14}"
mkdir -p logs output/random

# Caduceus-full: ready data already adapted; human genome ZS holdout
# M1=10 ep (TPM), M2=5 ep (predict M1 fold); outputs under output/random/
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

conda run -n caduceus_env python -m src.runs.caduceus_full \
  --strategy random \
  --raw raw \
  --ready ready \
  --out-root output/random \
  --seed 42 \
  --epochs-m1 10 \
  --epochs-m2 5 \
  --zs-genomes human \
  --nproc 4 \
  --batch-size 2 \
  --max-length 8192
