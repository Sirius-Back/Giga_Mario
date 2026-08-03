#!/usr/bin/env bash
# After Caduceus extract completes: pairwise + triangle heatmaps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${1:-results/embed_caduceus/pairwise}"
EMBED="${2:-results/embed_caduceus}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
conda run -n legnet --no-capture-output python -u -m src.embed.run_pairwise \
  --embed-root "$EMBED" \
  --out "$OUT" \
  --layers pooled,stage1_2,stage0,head_h \
  --role all \
  --max-n 4096 \
  --rdm-n 1536 \
  --seed 42 \
  --loo-fold 0 \
  2>&1 | tee "$OUT/run_pairwise.log"
conda run -n legnet --no-capture-output python -u -m src.embed.replot_pairwise \
  --out "$OUT" \
  --label-fontsize 20 \
  2>&1 | tee -a "$OUT/run_pairwise.log"
echo "Done → $OUT"
