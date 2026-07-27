#!/bin/bash
#SBATCH --job-name=caduceus_random
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/caduceus_random_%j.out
#SBATCH --error=logs/caduceus_random_%j.err

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate caduceus_env
cd /home/User14
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/train_caduceus_random_folds.py \
  --data-root data/caduceus_gb/random_full \
  --runs-dir runs/random \
  --epochs 10 \
  --batch-size 4 \
  --max-length 512 \
  --num-workers 2 \
  --seed 42
