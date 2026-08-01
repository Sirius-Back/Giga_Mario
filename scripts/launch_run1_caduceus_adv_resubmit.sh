#!/usr/bin/env bash
set -euo pipefail
cd /home/User14
exec conda run -n caduceus_env --no-capture-output \
  python -m src.runs_unif.run1_caduceus_random.continue_from_split \
  skip_direct=true force_adv=true
