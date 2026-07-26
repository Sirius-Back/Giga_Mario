#!/bin/bash
#SBATCH --job-name=ancombc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ancombc_%j.out
#SBATCH --error=logs/ancombc_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

RDS="${RDS:-}"
OUTDIR="${OUTDIR:-test/ancombc/slurm-run}"
TARGETS="${TARGETS:-}"
LEVELS="${LEVELS:-}"

EXTRA=()
if [ -n "$RDS" ]; then EXTRA+=(--rds "$RDS"); fi
if [ -n "$TARGETS" ]; then EXTRA+=(--targets "$TARGETS"); fi
if [ -n "$LEVELS" ]; then EXTRA+=(--levels "$LEVELS"); fi

Rscript .cursor/skills/ancombc/scripts/ancombc.R \
  --outdir "$OUTDIR" \
  "${EXTRA[@]}"
