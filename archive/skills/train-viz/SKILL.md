---
name: train-viz
description: >-
  Publication-quality training viz from run logs (one or compared models).
  Write/reuse src/train_viz, exec python -m src.train_viz. Use for /train-viz.
disable-model-invocation: true
---

# Train Viz

## Purpose

Produce publication-ready training visualizations (Nature Methods / NMI style) from
training logs — **zero manual editing**.

Follow: validation-first, missing-data-policy, reproducibility, publication-figures,
artifact-registry, scientific-integrity.

## Code-first contract (LOCKED)

```
@train-viz cycle:
  1. Collect log dirs / run dirs (from @caduceus → runs/.../logs)
  2. WRITE / UPDATE src/train_viz/ (do not reimplement plots in-chat)
  3. EXEC: python -m src.train_viz --models … -o figures/…
  4. REUSE the same package for later comparisons
```

Package path: `src/train_viz/` (symlink `src/train-viz` → `train_viz`).

## Input

| Input | Meaning |
|-------|---------|
| **logs** | Log files, globs, or directories containing `train_metrics.jsonl` / `metrics.log` |
| **models** | One run **or** a list of compared runs (`runs/caduceus/<name>` or short names) |

Auto-resolves `runs/<name>/logs/train_metrics.jsonl` (Caduceus layout).

## Output

Under `-o` / `--outdir` (default `figures/train-viz/`):

| Artifact | Role |
|----------|------|
| `Figure_XX_<metric>.{pdf,svg,png}` | Learning curves |
| `Figure_XX_multimodel_*` | Multi-model (when ≥2 models) |
| `Figure_XX_final_performance.*` | Final bars |
| `Figure_XX_generalization_gap.*` | val − train |
| `Figure_XX_early_stopping.*` | Best / patience |
| `manuscript/` | Key figure copies |
| `training_summary.csv` / `.md` | Tables |
| `visualization_config.yaml` | Reproducible style |

PNG @ **600 dpi**; PDF/SVG vector; Okabe–Ito / train=blue val=orange test=green.

## Exact command

```bash
# Single model (Caduceus run dir)
conda run -n caduceus_env python -m src.train_viz \
  --models runs/caduceus/smoke_M1 \
  -o figures/train-viz/smoke_M1 \
  --title "Caduceus M1 smoke"

# Compare models
python -m src.train_viz \
  --models runs/caduceus/M1 runs/caduceus/M2 \
  -o figures/train-viz/compare \
  --ribbon ci95

# Raw log paths
python -m src.train_viz \
  runs/caduceus/smoke_M1/logs/train_metrics.jsonl \
  -o figures/train-viz/smoke_M1
```

| Flag | Default | Notes |
|------|---------|-------|
| `--models` | — | One or many run dirs/names |
| positional / `--logs` | — | Files, globs, or log dirs |
| `-o` | `figures/train-viz` | Output root |
| `--ribbon` | `ci95` | Multi-seed ribbons |
| `--patience` | none | Early-stop annotation |
| `--dpi` | 600 | PNG |
| `--column` | `double` | or `single` |

## Style (locked)

See `src/train_viz/visualization_config.yaml` — Nature Methods palette, no matplotlib defaults,
no smoothing of stored values (optional LOWESS overlay only).

## Workflow checklist

```
train-viz:
- [ ] Resolve ≥1 log (file or run dir)
- [ ] Update src/train_viz only if behavior must change
- [ ] Exec python -m src.train_viz …
- [ ] Verify Figure_* + training_summary + config in outdir
- [ ] Register artifacts
```

## Coordination

| Skill / path | Role |
|--------------|------|
| `@caduceus` / `src/caduceus.py` | Produces `runs/.../logs/` |
| `src/train_viz/` | **Only** viz entry |
| publication-figures | Enforced by script QC |

## Additional resources

- [`src/train_viz/viz.py`](../../src/train_viz/viz.py)
- [`src/train_viz/visualization_config.yaml`](../../src/train_viz/visualization_config.yaml)
