#!/bin/bash
#SBATCH --job-name=difftree_ggdiffclade
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/difftree_ggdiffclade_%j.out
#SBATCH --error=logs/difftree_ggdiffclade_%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p logs

RDS="${RDS:-}"
OUTDIR="${OUTDIR:-test/difftree-ggdiffclade/slurm-run}"
TARGET="${TARGET:-}"

EXTRA=()
if [ -n "$RDS" ]; then EXTRA+=(--rds "$RDS"); fi
if [ -n "$TARGET" ]; then EXTRA+=(--target "$TARGET"); fi

Rscript .cursor/skills/difftree-ggdiffclade/scripts/difftree_ggdiffclade.R \
  --outdir "$OUTDIR" \
  "${EXTRA[@]}"
