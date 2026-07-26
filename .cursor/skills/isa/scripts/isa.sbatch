#!/bin/bash
#SBATCH --job-name=isa
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/isa_%j.out
#SBATCH --error=logs/isa_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

RDS="${RDS:-}"
OUTDIR="${OUTDIR:-test/isa/slurm-run}"
TARGET="${TARGET:-}"
TOPN="${TOPN:-30}"
NPERM="${NPERM:-9999}"

EXTRA=()
if [ -n "$RDS" ]; then EXTRA+=(--rds "$RDS"); fi
if [ -n "$TARGET" ]; then EXTRA+=(--target "$TARGET"); fi

Rscript .cursor/skills/isa/scripts/isa.R \
  --outdir "$OUTDIR" \
  --top-n "$TOPN" \
  --nperm "$NPERM" \
  "${EXTRA[@]}"
