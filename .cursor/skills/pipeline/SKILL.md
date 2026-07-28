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
| Fold-class rewrite | **Required:** `apply_fold_class_targets` after adversarial `split_predict` |

Optional: **`zsv=true`** → after each real train, eval final model on
`{out_root|adversarial}/PARSED|PREDICT/zero-shot-validation` via
`src.pipeline.zsv_eval` (fail if trees missing).

## Stage order

```
1. validate panel
2. /split (direct) → train (task_type) → optional ZSV eval
3. /adversarial copy
4. random split_predict
5. apply_fold_class_targets  # train/val/test → 0/1/2; ZSV keeps continuous
6. materialize SPLIT
7. /train adversarial (adversarial_task_type) → optional ZSV eval
```

Model CLIs are declared in `configs/train/{legnet,caduceus}.yaml`
(`direct_cmd`, `adversarial_cmd`, `zsv_cmd`).

## Workflow checklist

```
pipeline:
- [ ] Confirm Hydra overrides (run_id, mode, train, epochs, gpus, adversarial, zsv)
- [ ] Confirm panel_root complete
- [ ] dry: python -m src.hydra_pipeline mode=dry …
- [ ] run: python -m src.hydra_pipeline mode=run …
- [ ] Verify fold-class sidecar adversarial/PREDICT/predict_target.json when adversarial
- [ ] Verify logs/zero_shot_metrics.json when zsv=true
- [ ] method-decision + artifact-registry
```

## Coordination

| Skill / module | Role |
|----------------|------|
| `/split` | Fold assignment + materialize |
| `/train` | Fine-tune + optional ZSV |
| `/adversarial` | Copy + random split + **fold-class PREDICT** |
| `src.hydra_pipeline` | This orchestrator |
| `configs/` | Reproducible parameters + model commands |

## Rules

- Prefer Hydra over hand-rolled argparse orchestrators for new runs
- Reuse `./src` only
- `dry` never runs full training (`smoke=True` path)
- Register outputs in `docs/artifact-registry.md`
