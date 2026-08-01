#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
exec conda run -n legnet --no-capture-output \
  python -m src.runs_unif.run4_legnet_gc_kmeans_elbow.continue_from_split
