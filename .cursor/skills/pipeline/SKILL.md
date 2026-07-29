---
name: pipeline
description: >-
  End-to-end Hydra orchestrator (configs/pipeline.yaml): validate inputs,
  /split, /train, optional /adversarial (fold-class PREDICT) + adversarial
  /train, optional ZSV final-model eval. Re-runnable without agent. Use for /pipeline.
disable-model-invocation: true
---

# Pipeline (`/pipeline`)

**Preferred entry:** Hydra — `python -m src.hydra_pipeline` with
[`configs/pipeline.yaml`](../../../configs/pipeline.yaml) +
[`configs/train/*.yaml`](../../../configs/train/) (concrete model launch commands).

Legacy thin wrappers under `src/run/<run_id>/pipeline.py` may still exist for a
named run, but **new runs must use Hydra** so parameters and model CLIs are
reproducible (not ad-hoc `model_dir` on the orchestrator).

## Variants

| Variant | Behavior |
|---------|----------|
| **`mode=run`** | Execute stages end-to-end; monitor long jobs |
| **`mode=dry`** | Stage split / fold-class rewrite / smoke train logs; **no** full GPU train |

```bash
# dry
python -m src.hydra_pipeline mode=dry run_id=run0

# run (example)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.hydra_pipeline mode=run \
  run_id=run0 epochs=3 n_devices=4 train=legnet
```

Resolved configs + command templates are written to
`{out_root}/hydra_resolved_config.yaml` and `hydra_resolved_commands.yaml`.

### During-train monitoring (learning curves + TensorBoard)

Refresh anytime while / after training (reads live Lightning ``metrics.csv`` →
jsonl → cnsplots + Altair learning curves; refreshes split_compare; exports
``tensorboard/`` from train metrics). **Hydra `/pipeline` auto-runs**
``src.pipeline.pipeline_viz`` after train: train monitor + **SBS PCA** on
``split.csv`` (GC%/AAA%; works for ``random`` and ``gc``), via ``viz_conda_env``
(default ``caduceus_env``) when the train env lacks matplotlib.

```bash
# one-shot (single train outdir)
conda run -n caduceus_env python -m src.train_viz.train_monitor \
  --run-dir run/<run_id>/direct

# both direct + adversarial (when adversarial/train exists)
conda run -n caduceus_env python -m src.train_viz.train_monitor \
  --pipeline-root run/<run_id>

# poll every 60s while the job runs
watch -n 60 'conda run -n caduceus_env python -m src.train_viz.train_monitor \
  --run-dir run/<run_id>/direct --no-split-compare'

# full pipeline viz stage (train + SBS) — same entry Hydra calls
conda run -n caduceus_env python -m src.pipeline.pipeline_viz \
  --out-root runs/<run_id> --panel-root <panel> --train-dir runs/<run_id>/direct \
  --run-id <run_id>
```

TensorBoard (after sync / during train — **both** SummaryWriter + TensorBoardLogger):

```bash
tensorboard --logdir run/<run_id>/direct/tensorboard
# shows tensorboard/summary/ and tensorboard/lightning/
# adversarial (if trained):
tensorboard --logdir run/<run_id>/adversarial/train/tensorboard
```

Hydra `/train` (single stage) and `/pipeline` both launch LegNet/Caduceus via
`configs/train/*.yaml` + `src.pipeline.train.run_train` (dual TB finalized there):

```bash
python -m src.hydra_train mode=direct train=legnet run_id=run0 epochs=3
python -m src.hydra_pipeline mode=run train=caduceus run_id=run0
```

Outputs: `{run}/figures/train_monitor/Figure_*learning_curves*` (+ Altair HTML),
`{run}/figures/sbs/` PCA panels, `{run}/figures/split_compare/`,
synced `logs/train_metrics.jsonl`, and `{run}/tensorboard/` event files.
Hydra pipeline calls monitor after each `/train` and a final
`refresh_pipeline_monitors` (direct + adversarial when present).

### Split comparison figures

```bash
conda run -n caduceus_env python -m src.train_viz.split_compare \
  --run-dir run/<run_id>/direct \
  -o run/<run_id>/direct/figures/split_compare
```

Outputs: `split_metrics_compare.csv/.json`, `Figure_*_split_compare_train_val_test_zsv.{pdf,svg,png}`,
`Figure_*_altair.html` + `.vl.json`. Also invoked from `train_monitor` / pipeline after train
(including **adversarial/train** when that stage ran).

## Obligatory inputs (Hydra keys)

| Key | Meaning |
|-----|---------|
| **`run_id`** | Experiment id → `panel_root` / `out_root` default `run/${run_id}` |
| **`mode`** | `dry` \| `run` |
| **`data`** | Data panel id (metadata) |
| **`split`** | Split strategy (`random` for adversarial path) |
| **`train`** | Config group: `legnet` \| `caduceus` (`configs/train/`) |
| **`task_type`** | Direct train: `regression` \| `classification` |
| **`panel_root`** | Prepared panel (`ID.csv`, `PARSED`, `PREDICT`, optional `fold.csv`) |
| **`out_root`** | Artifact root |

When **`adversarial=true`** (default in pipeline.yaml):

| Key | Meaning |
|-----|---------|
| **`adversarial_task_type`** | Usually `classification` (fold-class 0/1/2) |
| Fold-class rewrite | **Required:** `apply_fold_class_targets(..., label_split_csv=<direct split.csv>)` after adversarial `split_predict` |

Optional: **`zsv=true`** → after each real train, eval **final_model** on
`{out_root|adversarial}/PARSED|PREDICT/zero-shot-validation` via
`src.pipeline.zsv_eval` (fail if trees missing).

**Universal ZSV contract** (LegNet + Caduceus):

| Piece | Role |
|-------|------|
| Trees | `{PARSED\|FASTA}/zero-shot-validation/*.ext` + matching `PREDICT/zero-shot-validation` |
| Loader | `src.pipeline.zsv_eval.load_zsv_pairs` |
| Metrics + artifacts | `metrics_from_preds` → `write_zsv_artifacts` → `logs/zero_shot_metrics.json` (+ jsonl/log) |
| Caduceus adapter | `src.caduceus.evaluate_zsv_root` (HF `final_model/`, same JSON shape) |
| LegNet adapter | `eval_legnet_zsv` (Lightning ckpt) |
| Viz | `train_monitor` + `split_compare` consume ZSV metrics automatically |

```bash
# Caduceus / LegNet (same CLI)
python -m src.pipeline.zsv_eval --model caduceus \
  --outdir runs/run1/direct --split-root runs/run1 --device 0
```

### Checkpoints (every 10 epochs → best as final)

| Behavior | Detail |
|----------|--------|
| **Periodic** | `checkpoint_every_n_epochs` (default **10**) → `checkpoints/epochN/` (Caduceus HF) or `epoch-*.ckpt` under `checkpoints/` (LegNet) |
| **Best** | Tracked live in `best_model/` (Caduceus: min val_loss; LegNet: max val_pearson) + `best_meta.json` |
| **Final** | After train, **`final_model/` = best checkpoint** (not last epoch). ZSV eval uses `final_model/`. |
| **Train viz** | Learning curves + early-stopping figures mark the selected final/best epoch as a ★ point (from `best_meta.json`) |

Override: `checkpoint_every_n_epochs=0` disables periodic dumps (best/final selection still runs).

## Stage order

```
1. validate panel
2. /split (direct) → train (task_type) → optional ZSV eval → **train_monitor + TensorBoard**
3. /adversarial copy
4. random split_predict
5. apply_fold_class_targets(label_split_csv=direct/previous split)  # prev train/val/test → 0/1/2
6. materialize SPLIT  # new adv folds; TRAIN mixes 0/1/2
7. /train adversarial (adversarial_task_type) → optional ZSV eval → **train_monitor + TensorBoard**
8. **refresh_pipeline_monitors** (direct + adversarial viz/TB if adversarial ran)
```

Model CLIs are declared in `configs/train/{legnet,caduceus}.yaml`
(`direct_cmd`, `adversarial_cmd`, `zsv_cmd`).

## Workflow checklist

```
pipeline:
- [ ] Confirm Hydra overrides (run_id, mode, train, epochs, gpus, adversarial, zsv, checkpoint_every_n_epochs)
- [ ] Confirm panel_root complete
- [ ] dry: python -m src.hydra_pipeline mode=dry …
- [ ] run: python -m src.hydra_pipeline mode=run …
- [ ] Verify fold-class sidecar adversarial/PREDICT/predict_target.json when adversarial
- [ ] Verify logs/zero_shot_metrics.json when zsv=true
- [ ] Verify `checkpoints/` periodic dumps every N epochs (when N>0 and epochs≥N)
- [ ] Verify `best_model/best_meta.json` and `final_model/` == selected best
- [ ] Verify `{direct,adversarial/train}/figures/train_monitor/` learning curves mark final/best ★
- [ ] Verify `{direct,adversarial/train}/figures/split_compare/` (cnsplots + Altair) when metrics exist
- [ ] Verify `{direct,adversarial/train}/tensorboard/` has train metrics event files
- [ ] method-decision + artifact-registry
```

## Coordination

| Skill / module | Role |
|----------------|------|
| `/split` | Fold assignment + materialize |
| `/train` | Fine-tune + optional ZSV |
| `/adversarial` | Copy + random split + **fold-class PREDICT** |
| `src.hydra_pipeline` | This orchestrator |
| `src.hydra_train` | Hydra `/train` entry (LegNet/Caduceus) |
| `src.pipeline.zsv_eval` | Universal ZSV dispatch (`load_zsv_pairs` + model adapters) |
| `src.caduceus.evaluate_zsv_root` | Caduceus HF ZSV adapter |
| `src.tb_logging` | Dual SummaryWriter + TensorBoardLogger helpers |
| `src.train_viz.train_monitor` | Sync Lightning→jsonl + learning-curve monitor figs + TB export |
| `src.train_viz.tensorboard_metrics` | Caduceus-shaped `tensorboard/` from train jsonl |
| `src.train_viz.split_compare` | train/val/test/ZSV metric bars (cnsplots + Altair) |
| `configs/` | Reproducible parameters + model commands |

## Rules

- Prefer Hydra over hand-rolled argparse orchestrators for new runs
- Reuse `./src` only
- `dry` never runs full training (`smoke=True` path)
- Register outputs in `docs/artifact-registry.md`
