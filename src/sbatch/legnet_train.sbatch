#!/bin/bash
#SBATCH --job-name=legnet_ft
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/legnet_%j.out
#SBATCH --error=logs/legnet_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

DATA_PATH="${DATA_PATH:?Set DATA_PATH to legnet_ready TSV}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-777}"
DEVICE="${DEVICE:-0}"
OUT="${OUT:-}"
DEMO="${DEMO:-1}"

EXTRA=(--use-shift --reverse-augment)
if [[ -n "${OUT}" ]]; then
  EXTRA+=(--out "${OUT}")
fi
if [[ "${DEMO}" == "1" ]]; then
  EXTRA+=(--demo)
fi

conda run -n legnet --no-capture-output python -m src.legnet \
  --data-path "${DATA_PATH}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}"
