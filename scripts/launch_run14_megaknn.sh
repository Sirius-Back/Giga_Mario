#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
exec conda run -n caduceus_env --no-capture-output \
  python -m src.runs_unif.run14_caduceus_kmer_k7.continue_from_split \
  force_resplit=true max_fold_size=2000
