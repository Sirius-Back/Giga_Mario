#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
exec conda run -n caduceus_env --no-capture-output \
  python -m src.runs_unif.run20_caduceus_pangenome_k10_w0_100.continue_from_split
