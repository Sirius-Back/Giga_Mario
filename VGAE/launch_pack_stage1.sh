#!/usr/bin/env bash
set -eo pipefail
cd /home/User14
# Avoid `set -u`: conda activate scripts reference unset MKL_INTERFACE_LAYER.
source /home/User14/miniconda3/etc/profile.d/conda.sh
conda activate caduceus_env
export PYTHONPATH=/home/User14
exec python -u src/run/run_id/pack_vgae_stage1.py
