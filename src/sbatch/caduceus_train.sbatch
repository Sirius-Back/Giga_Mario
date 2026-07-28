#!/bin/bash
#SBATCH --job-name=caduceus_ft
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/caduceus_%j.out
#SBATCH --error=logs/caduceus_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

SPLITS_DIR="${SPLITS_DIR:-splits/random/M1}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
OUT="${OUT:-}"

EXTRA=()
if [[ -n "${OUT}" ]]; then
  EXTRA+=(--out "${OUT}")
fi

torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
  -m src.caduceus \
  --splits-dir "${SPLITS_DIR}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --batch-size 16 \
  --max-length 2048 \
  --amp \
  "${EXTRA[@]}"
