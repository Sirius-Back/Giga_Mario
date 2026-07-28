#!/usr/bin/env bash
# Caduceus fine-tune: ready_splits/random/{M1,M2} → clean/full_genes_transcriptome/random/{M1,M2}
# Sequential: M1 (TPM regression) then M2 (fold-class classification).
#
# Speed defaults (4×V100-32GB probe, 2026-07-28):
#   batch=16, max_length=2048, --amp
#   (batch 16 @ 8192 OOMs; larger max_length is slower, not faster)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TORCHRUN="${ROOT}/miniconda3/envs/caduceus_env/bin/torchrun"
NGPU="${NGPU:-4}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
BATCH="${BATCH:-16}"
MAXLEN="${MAXLEN:-2048}"
AMP="${AMP:-1}"
LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}" clean/full_genes_transcriptome/random

run_one() {
  local model="$1"
  local splits="${ROOT}/ready_splits/random/${model}"
  local out="${ROOT}/clean/full_genes_transcriptome/random/${model}"
  local log="${LOGDIR}/caduceus_ready_v2_${model}.log"
  local amp_flags=()
  if [[ "${AMP}" == "1" || "${AMP}" == "true" ]]; then
    amp_flags+=(--amp)
  fi
  echo "[$(date -Is)] START ${model} splits=${splits} out=${out} batch=${BATCH} max_length=${MAXLEN} amp=${AMP}" | tee -a "${log}"
  "${TORCHRUN}" --standalone --nproc_per_node="${NGPU}" \
    -m src.caduceus \
    --splits-dir "${splits}" \
    --out "${out}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --batch-size "${BATCH}" \
    --max-length "${MAXLEN}" \
    "${amp_flags[@]}" \
    2>&1 | tee -a "${log}"
  echo "[$(date -Is)] DONE ${model} exit=$?" | tee -a "${log}"
}

run_one M1
run_one M2
echo "[$(date -Is)] ALL DONE" | tee -a "${LOGDIR}/caduceus_ready_v2_pipeline.log"
