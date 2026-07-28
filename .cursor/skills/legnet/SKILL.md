---
name: legnet
description: >-
  human_legnet train/eval: write/reuse src/legnet.py, exec on legnet_ready TSV
  (230 bp), log Caduceus-like train_metrics.jsonl for @train-viz, save
  checkpoints. Use for /legnet after @legnet-adapt.
disable-model-invocation: true
---

# LegNet

MPRA-LegNet / human_legnet (Agarwal et al.; Penzar et al.) — CNN for short
regulatory DNA (230 bp lentiMPRA oligo format).

- Repo: https://github.com/autosome-ru/human_legnet
- Upstream: https://github.com/autosome-ru/LegNet
- Vendor pin: `software/human_legnet` (see `method-decision.md`)
- Conda: `legnet` (`software/human_legnet/envs/environment.yml`)

## Code-first contract (LOCKED)

```
@legnet cycle:
  1. Ensure TSV exists (from @legnet-adapt → legnet_ready/*.tsv, folds 1..10)
  2. WRITE / UPDATE src/legnet.py (do not reimplement training in-chat)
  3. EXEC: python -m src.legnet --data-path <tsv> …
  4. REUSE the same script for later runs / GPUs / seeds
```

**Never** invent a parallel trainer in-chat. Extend `src/legnet.py` instead.
Upstream train loop stays in `software/human_legnet/core.py`; this module
validates I/O, launches it, and normalizes logs for `@train-viz`.

Prep: `@legnet-adapt` / `src/legnet_preprocess.py`. Metrics summary helper:
`src/legnet_demo_metrics.py`.

## Input / output

| | Path |
|--|------|
| **Input** | human_legnet TSV: `seq_id seq mean_value fold rev` (folds **1..10**) |
| **Output** | `--out` (default `runs/legnet/<data_stem>/`) |

```
OUT/
  logs/              run_config.json, train_metrics.jsonl, metrics.log, epoch{N}/
  config.json        human_legnet TrainingConfig dump
  model_{val}_{test}/  Lightning logs, checkpoints, predictions_new_format.tsv
  best_model/        best val_pearson .ckpt (copy)
  final_model/       last epoch .ckpt (copy)
  metrics_summary.json / .md
  train_time.json
```

Regression target: continuous **`mean_value`** (project: TPM from `@legnet-adapt`).
Native human_legnet logs `val_loss` / `val_pearson` each epoch; wrapper also
emits Caduceus-shaped `train_metrics.jsonl` for `@train-viz`.

## Exact command

```bash
# Demo (1 CV split: test=1, val=2) — recommended smoke
conda run -n legnet python -m src.legnet \
  --data-path legnet_ready/GCF_000001405.40_folds1to10.tsv \
  --out runs/legnet/demo_GRCh38 \
  --epochs 20 --device 0 --demo \
  --use-shift --reverse-augment

# Full 10-fold CV (90 models) — long
conda run -n legnet python -m src.legnet \
  --data-path legnet_ready/all.tsv \
  --out runs/legnet/all_cv \
  --epochs 20 --device 0 \
  --use-shift --reverse-augment
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data-path` | required | `legnet_ready` TSV |
| `--out` | `runs/legnet/<stem>/` | logs + ckpts |
| `--epochs` | 20 | per `model-train.mdc` |
| `--device` | 0 | Primary GPU (single-GPU mode) |
| `--n-devices` | 1 | GPUs via Lightning `ddp_spawn` when >1 |
| `--seed` | 777 | human_legnet default |
| `--demo` | off | one split only (test=1, val=2) |
| `--use-shift` / `--reverse-augment` | off | match paper demo |
| `--train-batch-size` | 1024 | |
| `--vendor` | `software/human_legnet` | core.py root |

SLURM: `src/sbatch/legnet_train.sbatch` (runs `python -m src.legnet`).

## Metrics

**During train (Lightning):** `train_loss`, `val_loss`, `val_pearson`.

**After train (`src/legnet_demo_metrics.py`):** on test predictions
`mean(forw_pred,rev_pred)` vs `mean_value`:

```text
pearson, spearman, mse, rmse, mae, r2
```

**For `@train-viz`:** `OUT/logs/train_metrics.jsonl` with per-epoch
`train.loss`, `validation.loss`, `validation.pearson`.

Caveat: labels from `@legnet-adapt` are RNA-seq TPM, not lentiMPRA activity —
do not over-interpret vs published MPRA-LegNet numbers.

## Defaults / process

Follow [`.cursor/rules/model-train.mdc`](../../rules/model-train.mdc):

- Validate TSV columns + fold ∈ {1..10} before launch
- Seed, checkpoint (`final_model/` + `best_model/` on val_pearson)
- Register artifacts; update `method-decision.md` for nontrivial choices
- Long jobs → SLURM + `@monitor` (10 min after submit)

## Vendor / env

| Path | Role |
|------|------|
| `software/human_legnet` | Primary trainer (`core.py`) |
| `software/LegNet` | Upstream yeast/DREAM reference |
| conda `legnet` | Runtime |

## Coordination

| Skill / path | Role |
|--------------|------|
| `@legnet-adapt` / `src/legnet_preprocess.py` | Build `legnet_ready/` TSV + BED |
| `src/legnet.py` | **Only** training entry |
| `src/legnet_demo_metrics.py` | Post-hoc test metrics summary |
| `metrics.md` | Regression metric vocabulary (test summary) |
| `@train-viz` | Plots from `OUT/logs/train_metrics.jsonl` |

## Workflow checklist

```
legnet:
- [ ] Confirm --data-path TSV (seq_id,seq,mean_value,fold,rev; folds 1..10)
- [ ] Confirm software/human_legnet + conda env legnet
- [ ] Update src/legnet.py if behavior must change
- [ ] Exec python -m src.legnet --data-path …
- [ ] Verify logs/train_metrics.jsonl + best_model/ + metrics_summary.md
- [ ] method-decision + artifact-registry
- [ ] Optional: python -m src.train_viz --models <OUT> -o figures/train-viz/…
```

## Additional resources

- [wiki/legnet_conversion.md](../../wiki/legnet_conversion.md)
- [README.md](README.md)
- Project entry: [`src/legnet.py`](../../src/legnet.py)
