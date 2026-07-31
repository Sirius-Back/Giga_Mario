#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
exec conda run -n caduceus_env --no-capture-output \
  python -m src.runs_unif.run22_caduceus_pangenome_k10_wm100_100.continue_from_split
