#!/bin/bash
#SBATCH --job-name=network
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/network_%j.out
#SBATCH --error=logs/network_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

RDS="${RDS:-}"
OUTDIR="${OUTDIR:-test/network/slurm-run}"
METHOD="${METHOD:-netcomi}"
MEASURE="${MEASURE:-sparcc}"
MIN_MEAN_REL="${MIN_MEAN_REL:-1e-4}"
LABEL_MODE="${LABEL_MODE:-shadowtext}"
# Optional cap; omit TOPN (or set empty) for abundance-filter-only
TOPN="${TOPN:-}"

EXTRA=()
if [ -n "$RDS" ]; then EXTRA+=(--rds "$RDS"); fi
if [ -n "$TOPN" ]; then EXTRA+=(--top-n "$TOPN"); fi

Rscript .cursor/skills/network/scripts/network.R \
  --outdir "$OUTDIR" \
  --method "$METHOD" \
  --measure "$MEASURE" \
  --min-mean-rel "$MIN_MEAN_REL" \
  --label-mode "$LABEL_MODE" \
  "${EXTRA[@]}"
