#!/usr/bin/env bash
# Reusable @adapt entry (T-4 writes; T-5/T-8 reuse). Project-relative paths.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec conda run -n caduceus_env python scripts/adapt.py \
  --config scripts/adapt_config.default.yaml \
  "$@"
