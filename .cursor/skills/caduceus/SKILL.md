---
name: caduceus
description: >-
  Caduceus DNA LM fine-tune/eval: write/reuse src/caduceus.py, exec on a splits
  directory, log metrics.md + TensorBoard, save final_model. Use for /caduceus.
disable-model-invocation: true
---

# Caduceus

Caduceus (Schiff et al., 2024) — bi-directional RC-aware DNA LM (Mamba/SSM).

- Repo: https://github.com/kuleshov-group/caduceus
- HF: https://huggingface.co/collections/kuleshov-group/caducues-65dcb89b4f54e416ef61c350
- Vendor pin: `software/caduceus` (see `method-decision.md`)

## Code-first contract (LOCKED)

```
@caduceus cycle:
  1. Ensure folds exist (from @split → e.g. splits/random/M1)
  2. WRITE / UPDATE src/caduceus.py (do not reimplement training in-chat)
  3. EXEC: python -m src.caduceus --splits-dir <dir> …
  4. REUSE the same script for later runs / GPUs / seeds
```

**Never** invent a parallel trainer in-chat. Extend `src/caduceus.py` instead.

Metrics helpers: `src/metrics_logging.py` (imported by `src/caduceus.py`; skill `scripts/metrics_logging.py` is a thin re-export).

## Input / output

| | Path |
|--|------|
| **Input** | Directory with `train|val|test/{sequences/*.txt, labels.tsv}` (e.g. `splits/random/M1`) |
| **Output** | `--out` (default `runs/caduceus/<splits_name>/`) |

```
OUT/
  logs/           run_config.json, train_metrics.jsonl, metrics.log, epoch{N}/
  tensorboard/    TensorBoard event files
  final_model/    HF checkpoint + tokenizer
  best_model/     best val_loss checkpoint
  train_time.json
```

Task auto-detect:

- `labels.tsv` has `label` → **classification** (M2)
- else `TPM` → **regression** (M1) with full **`metrics.md`** suite each epoch

## Exact command

```bash
# Single GPU
conda run -n caduceus_env python -m src.caduceus \
  --splits-dir splits/random/M1 \
  --epochs 20 --seed 42 --max-length 8192

# Multi-GPU
torchrun --standalone --nproc_per_node=4 -m src.caduceus \
  --splits-dir splits/random/M1 \
  --epochs 20 --seed 42

# TensorBoard
tensorboard --logdir runs/caduceus/random_M1/tensorboard
```

| Flag | Default | Notes |
|------|---------|-------|
| `--splits-dir` | required | Fold root |
| `--out` | `runs/caduceus/<name>/` | logs + tensorboard + final_model |
| `--epochs` | 20 | per `model-train.mdc` |
| `--max-length` | 8192 | truncates longer windows |
| `--task` | `auto` | `regression` \| `classification` |
| `--max-samples` | none | smoke-test cap per fold |

SLURM: `src/sbatch/caduceus_train.sbatch` (runs `python -m src.caduceus`).

## Metrics (required for regression)

Every epoch on train / validation / test (`metrics.md` via TorchMetrics):

```text
loss, pearson, spearman, mse, rmse, mae, r2,
genewise_pearson_median, samplewise_pearson_median
```

Also written to **TensorBoard** under `{split}/{metric}` and to `logs/epoch{N}/`.

Classification (M2): log `loss` + `accuracy` (+ TensorBoard).

## Defaults / process

Follow [`.cursor/rules/model-train.mdc`](../../rules/model-train.mdc):

- Validate splits before launch
- Seed, checkpoint (`final_model/` + `best_model/` on val_loss)
- Register artifacts; update `method-decision.md` for nontrivial choices
- Long jobs → SLURM + `@monitor` (10 min after submit)

## Checkpoints (HF)

| Checkpoint | RC |
|---|---|
| `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` | Ph — average fwd+RC at inference |
| `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` | PS — RC-equivariant (default in script) |

Always `trust_remote_code=True`. Tokenizer = 1 token / bp.

## Coordination

| Skill / path | Role |
|--------------|------|
| `@split` / `src/splits/` | Produce `--splits-dir` |
| `@adapt` / `src/preprocessing.py` | Build `ready/` before split |
| `src/caduceus.py` | **Only** training entry |
| `metrics.md` | Regression metric contract |
| `@train-viz` | Plots from `logs/` |

## Workflow checklist

```
caduceus:
- [ ] Confirm splits-dir has train/val/test + labels.tsv
- [ ] Update src/caduceus.py if behavior must change
- [ ] Exec python -m src.caduceus --splits-dir … 
- [ ] Verify logs/ + tensorboard/ + final_model/
- [ ] method-decision + artifact-registry
```
