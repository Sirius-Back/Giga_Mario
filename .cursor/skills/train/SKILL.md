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
| **`outdir`** | Run output root (`logs/`, `tensorboard/`, `final_model/`, figures) |

Optional but required when user/spec requests them:

| Input | Meaning |
|-------|---------|
| **`zero-shot-validation` / ZSV** | Path to ZSV trees from `@split` / `src.pipeline.split` (`…/zero-shot-validation/`) — eval **final model** when specified |
| **`epochs`**, **`seed`**, **`max_samples`** | Defaults: epochs per `model-train.mdc` (20), seed `42` |
| **`metrics.md`** | Regression metric contract (via Caduceus / TorchMetrics) |

If any obligatory input is missing → **stop** (missing-data-policy). Do not invent folds, labels, or metrics.

## Code-first contract (LOCKED)

```
/train cycle:
  1. Validate obligatory inputs + folders (train/val/test or SPLIT/FASTA+PREDICT)
  2. WRITE src/run/<run_id>/{data}_{split}_{train}_{direct|adversarial}.py
     — thin orchestrator: import src.pipeline.train / src.caduceus / src.legnet /
       src.pipeline.train_viz / src.train_viz; no parallel trainer
  3. EXEC that file (or python path/to/script.py)
  4. REUSE the same script for later runs / GPUs / seeds
```

**Never** invent a parallel trainer or plotter in-chat. Extend `src/pipeline/train.py`, `src/caduceus.py`, `src/legnet.py`, or `src/train_viz/` instead.

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
2. **TensorBoard** — Caduceus already writes `outdir/tensorboard/` via `SummaryWriter`. Do not bypass; reuse that path. LegNet: wire TB only through existing trainer hooks if present; do not invent metrics.
3. **metrics.md** — regression epochs must log the full suite (via Caduceus / `src.metrics_logging`).
4. **Visualization** — call `src.pipeline.train_viz.run_train_viz` and/or `src.train_viz` on the run `logs/` (publication figures under `outdir/figures/` or similar).
5. **Zero-shot-validation** — if ZSV is specified and trees exist from split:
   - Evaluate the **final model** on ZSV (same metric keys as validation when regression).
   - Log results to `logs/` + TensorBoard under a `zero-shot-validation/…` tag when TB is active.
   - If ZSV requested but trees missing → **stop** (do not skip silently).
   - If ZSV not specified → omit ZSV eval (do not invent holdouts).

## Exact patterns

```bash
# Prefer exec of the written run script
conda run -n caduceus_env python src/run/<run_id>/<data>_<split>_<train>_direct.py

# Underlying stage (for smoke / debug only — still prefer the run script)
conda run -n caduceus_env python -m src.pipeline.train \
  --model caduceus --type regression \
  --folders <SPLIT> --outdir <outdir> \
  --epochs 20 --seed 42

# Viz
python -m src.pipeline.train_viz --logs <outdir> --outdir <outdir>/figures
# or
python -m src.train_viz --models <outdir> -o <outdir>/figures
```

## Workflow checklist

```
train:
- [ ] Confirm run_id, data, split, train model, mode (direct|adversarial)
- [ ] Confirm folders + type; ZSV path if requested
- [ ] Write src/run/<run_id>/{data}_{split}_{train}_{mode}.py (imports only)
- [ ] Exec that script
- [ ] Verify logs/ + tensorboard/ + final_model/ (+ figures/; + ZSV metrics if requested)
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
