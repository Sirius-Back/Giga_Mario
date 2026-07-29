---
name: train
description: >-
  Fine-tune Caduceus or LegNet on SPLIT trees: write/exec
  src/run/<run_id>/{data}_{split}_{train}_{direct|adversarial}.py reusing
  src.pipeline.train, src.caduceus, src.legnet, src.train_viz, metrics.md,
  TensorBoard, optional zero-shot-validation. Use for /train.
disable-model-invocation: true
---

# Train (`/train`)

Write-and-exec adapter for model fine-tune on pipeline **SPLIT** trees (or legacy Caduceus folds). Based on `@caduceus` / `@legnet` / `@train-viz` contracts — **must reuse `./src`**, never reimplement trainers or plots in-chat.

## Obligatory inputs

| Input | Meaning |
|-------|---------|
| **`run_id`** | Run directory under `src/run/<run_id>/` |
| **`data`** | Data panel id (filename stem; path to SPLIT or adapted folds) |
| **`split`** | Split strategy id (e.g. `random`) |
| **`train`** | Model key: `caduceus` \| `legnet` \| `human_legnet` |
| **`mode`** | `direct` \| `adversarial` (encoded in script filename) |
| **`folders`** | Materialized SPLIT root (or Caduceus `train/val/test` layout) |
| **`type`** | Task: `regression` (TPM / `predict_var1`) or `classification` |
| **`outdir`** | Run output root (`logs/`, `tensorboard/`, `checkpoints/`, `best_model/`, `final_model/` = best, figures) |

Optional but required when user/spec requests them:

| Input | Meaning |
|-------|---------|
| **`zero-shot-validation` / ZSV** | Path to ZSV trees from `@split` / `src.pipeline.split` (`…/zero-shot-validation/`) — eval **final model** (best checkpoint) when specified |
| **`epochs`**, **`seed`**, **`max_samples`** | Defaults: epochs per `model-train.mdc` (20), seed `42` |
| **`checkpoint_every_n_epochs`** | Default **10**; periodic dumps under `checkpoints/`; `0` disables |
| **`metrics.md`** | Regression metric contract (via Caduceus / TorchMetrics) |

If any obligatory input is missing → **stop** (missing-data-policy). Do not invent folds, labels, or metrics.

## Code-first contract (LOCKED)

```
/train cycle:
  1. Validate obligatory inputs + folders (train/val/test or SPLIT/FASTA+PREDICT)
  2. Prefer Hydra: python -m src.hydra_train mode=direct|adversarial train=legnet|caduceus …
     OR WRITE src/run/<run_id>/{data}_{split}_{train}_{direct|adversarial}.py
     — thin orchestrator calling src.hydra_train / src.pipeline.train; no parallel trainer
  3. EXEC Hydra or that file
  4. REUSE the same Hydra overrides / script for later runs / GPUs / seeds
```

**Never** invent a parallel trainer or plotter in-chat. Extend `src/pipeline/train.py`, `src/caduceus.py`, `src/legnet.py`, `src/tb_logging.py`, or `src/train_viz/` instead.

Follow project rules: skills-write-and-exec-src, model-train, validation-first, missing-data-policy, reproducibility, artifact-registry.

## Script path

```
src/run/<run_id>/{data}_{split}_{train}_{direct|adversarial}.py
```

Examples:

- `src/run/exp01/prok_random_caduceus_direct.py`
- `src/run/exp01/prok_random_caduceus_adversarial.py`

## What the run script must do

1. **Train** via `src.pipeline.train.run_train(...)` (dispatches to `src.caduceus` / `src.legnet`).
2. **TensorBoard (both loggers, both models)** — Always under `outdir/tensorboard/`:
   - `summary/` — `torch.utils.tensorboard.SummaryWriter`
   - `lightning/` — Lightning `TensorBoardLogger` (or SummaryWriter stand-in if PL import fails)
   Caduceus and LegNet both write both during train; `run_train` also backfills from jsonl. Do not invent metrics.
3. **metrics.md** — regression epochs must log the suite (loss, pearson, spearman, mse, rmse, mae, r2; genewise/samplewise when gene axes exist) via Caduceus / TorchMetrics / LegNet LitModel val+train collections. Classification: loss+accuracy (Caduceus).
4. **Visualization** — `run_train` calls `refresh_train_monitor` (learning curves + split_compare). Also `src.pipeline.train_viz` / `src.train_viz` as needed.
5. **Zero-shot-validation** — if ZSV is specified / `eval_zsv=True`:
   - Require `{zsv_root}/PARSED/zero-shot-validation` and `…/PREDICT/zero-shot-validation`.
   - Evaluate the **final model** via `src.pipeline.zsv_eval` (LegNet checkpoint + continuous ZSV labels; Caduceus when helper exists).
   - Write `logs/zero_shot_metrics.json` + append `metrics.log` / `train_metrics.jsonl`.
   - If ZSV requested but trees missing → **stop** (do not skip silently).
   - If ZSV not specified → omit ZSV eval.

## Exact patterns

```bash
# Preferred: Hydra /train
python -m src.hydra_train mode=direct train=legnet run_id=run0 epochs=3
python -m src.hydra_train mode=adversarial train=caduceus run_id=run0 zsv=true

# Or full pipeline (Hydra)
python -m src.hydra_pipeline mode=run train=legnet zsv=true

# Thin write-and-exec script (imports run_train / hydra_train)
conda run -n caduceus_env python src/run/<run_id>/<data>_<split>_<train>_direct.py
```
## Workflow checklist

```
train:
- [ ] Confirm run_id, data, split, train model, mode (direct|adversarial)
- [ ] Confirm folders + type; ZSV path if requested
- [ ] Write src/run/<run_id>/{data}_{split}_{train}_{mode}.py (imports only)
- [ ] Exec that script
- [ ] Verify logs/ + tensorboard/ + checkpoints/ + best_model/ + final_model/ (=best) (+ figures with best ★; + ZSV metrics if requested)
- [ ] method-decision + artifact-registry
```

## Coordination

| Module | Role |
|--------|------|
| `src.pipeline.train` | SPLIT → Caduceus/LegNet adapter + dispatch |
| `src.caduceus` | Caduceus train; **TensorBoard** |
| `src.legnet` | LegNet train |
| `src.pipeline.train_viz` / `src.train_viz` | Curves / summaries |
| `metrics.md` | Regression metric contract |
| `src.pipeline.split` | Produces SPLIT + optional ZSV trees |
| `/adversarial` | Builds adversarial panel before `mode=adversarial` train |

## Rules

- Reuse `./src` only — no in-chat reimplementation
- Filename **must** end with `_direct.py` or `_adversarial.py`
- Fixed seeds; relative project paths
- Register outputs in `docs/artifact-registry.md`
