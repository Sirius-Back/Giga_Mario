# runs_aligned/

Aligned reproducibility suite for Hydra `/pipeline` **rerun** jobs.

Every run here uses the same schedule defaults unless overridden:

| Setting | Default |
|---------|---------|
| `rerun` | `true` |
| `epochs` | `30` (max) |
| `min_epochs` | `10` |
| `early_stopping_patience` | `10` (early stop allowed) |
| epoch eval caps | `8192` train / val / test |
| new split ratios | train:test:val ≈ **3:1:1** only |
| overwrite | **forbidden** (source runs and existing outdirs) |

## Reuse a prior split (typical)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.hydra_pipeline mode=run \
  rerun=true \
  source_split=runs/run3 \
  run_id=run3_caduceus_aligned \
  train=caduceus \
  panel_root=ready_caduceus \
  split=gc \
  adversarial=false
```

`source_split` may be a prior run root or a path to `split.csv`. The prior
tree is **not** modified; folds are copied into `runs_aligned/<run_id>/` and
`SPLIT/` is rematerialized there.

## New 3:1:1 folds (no source)

```bash
python -m src.hydra_pipeline mode=dry \
  rerun=true \
  run_id=random_aligned_smoke \
  train=caduceus \
  panel_root=ready_caduceus \
  split=random \
  adversarial=false
```

Do not pass other ratio schedules under `rerun=true` without `source_split`.

See `.cursor/skills/pipeline/SKILL.md` (variant **rerun**).
