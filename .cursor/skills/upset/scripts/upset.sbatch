#!/bin/bash
#SBATCH --job-name=upset
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/upset_%j.out
#SBATCH --error=logs/upset_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

RDS="${RDS:-}"
OUTDIR="${OUTDIR:-test/upset/slurm-run}"
TARGET="${TARGET:-}"
FACTORS="${FACTORS:-}"
MINSIZE="${MINSIZE:-3}"

EXTRA=()
if [ -n "$RDS" ]; then EXTRA+=(--rds "$RDS"); fi
if [ -n "$TARGET" ]; then EXTRA+=(--target "$TARGET"); fi
if [ -n "$FACTORS" ]; then EXTRA+=(--factors "$FACTORS"); fi

Rscript .cursor/skills/upset/scripts/upset.R \
  --outdir "$OUTDIR" \
  --min-size "$MINSIZE" \
  "${EXTRA[@]}"
