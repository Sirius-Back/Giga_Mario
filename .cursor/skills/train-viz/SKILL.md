---
name: train-viz
description: >-
  Publication-quality training visualization (Nature Methods / NMI / NeurIPS /
  ICML): learning curves, multi-model/seed aggregates, final bars, gap/early-stop
  figures from one or more logs. Use for /train-viz or manuscript train plots.
disable-model-invocation: true
---

# Train Viz

## Purpose

Produce publication-ready ("Nature Machine Intelligence", "Nature Methods", NeurIPS, ICML) visualizations of the complete training process.

The output must require **zero manual editing** before insertion into a manuscript or supplementary materials.

Never produce matplotlib default figures.

Follow: **validation-first**, **missing-data-policy**, **reproducibility**,
**publication-figures**, **artifact-registry**, **scientific-integrity**.

## Extension: Publication-quality Training Visualization

### Goal

Produce publication-ready ("Nature Machine Intelligence", "Nature Methods", NeurIPS, ICML) visualizations of the complete training process.

The output must require **zero manual editing** before insertion into a manuscript or supplementary materials.

Never produce matplotlib default figures.

### Input

Input may contain one or multiple training runs.

Each run contains one or more of:

- train metrics
- validation metrics
- test metrics (optional, if evaluated every epoch)
- final evaluation metrics
- metadata
    - model name
    - random seed
    - split
    - fold
    - species
    - learning rate
    - batch size
    - optimizer
    - scheduler

Automatically infer available metrics.

Typical metrics:

- loss
- Pearson
- Spearman
- MSE
- RMSE
- MAE
- R²
- gene-wise Pearson
- sample-wise Pearson

Also accepted when present in logs: accuracy, f1, auc/auroc, precision, recall, lr / learning_rate, elapsed_sec.

### General style

Style:

Nature Methods / Nature Machine Intelligence.

Figure size:

single column:
1800×1400 px

double column:
3600×2400 px

vector version:

PDF
SVG

raster version:

PNG
600 dpi

Font

Arial

or

Helvetica

Font sizes

axis labels
9 pt

ticks
8 pt

legend
8 pt

title
10 pt

Line width

2.2

Grid

major only

light gray

alpha 0.3

No chartjunk.

No unnecessary borders.

Top/right spines removed.

Consistent color palette everywhere.

Use colorblind-safe palette.

### Color convention

Never randomly choose colors.

Always use

Train
blue

Validation
orange

Test
green

If multiple models

Each model receives one color

Train/Validation/Test are distinguished by line style

train
solid

validation
dashed

test
dotted

If several seeds

Plot mean

plus

95% confidence interval

or

standard deviation ribbon.

### Smoothing

Never smooth the actual stored values.

If smoothing requested:

display raw curve

+

LOWESS overlay.

Never hide noisy learning behaviour.

### Randomness

Whenever multiple random seeds exist

always compute

mean

median

standard deviation

95% confidence interval

Visualize uncertainty using ribbons.

Never show only the best run.

### Layout

Automatically determine optimal panel arrangement.

Preferred layouts

2 metrics

1×2

4 metrics

2×2

6 metrics

2×3

8 metrics

2×4

9 metrics

3×3

10+

multiple pages

Maintain identical axis sizes.

Perfect alignment.

Shared legends whenever possible.

### Per-metric learning curves

For every metric

produce

Epoch vs Metric

Include

train

validation

test (if available)

Mark

best epoch

using

large filled circle

Add

vertical dashed line

at

early stopping epoch

if applicable.

Annotate

best value

next to marker.

Metrics where lower is better

Loss

MSE

RMSE

MAE

Metrics where higher is better

Pearson

Spearman

R²

Gene-wise Pearson

Sample-wise Pearson

Automatically reverse "best" criterion.

### Multi-model comparison

If multiple models

Create separate figure for every metric.

Each figure contains

one panel

multiple colored curves

One line

=

one model

Separate figures for

Train

Validation

Test

Do NOT mix train and validation in the same plot for multi-model comparison.

### Final performance

Create publication-quality comparison plots.

Preferred

horizontal barplots

ordered

best→worst

Include

error bars

if multiple seeds.

Annotate

exact numeric values.

Metrics

Pearson

Spearman

RMSE

MAE

R²

Gene-wise Pearson

Sample-wise Pearson

Loss

Automatically orient

higher is better

or

lower is better.

### Seed variability

If multiple seeds

produce additionally

boxplots

or

violin plots

of final metrics.

Overlay individual runs.

Display

mean

median

outliers.

### Correlation of metrics

If enough epochs

produce

Spearman correlation matrix

between all tracked metrics.

Useful for

understanding convergence.

### Generalization gap

Automatically compute

validation - train

for every metric.

Produce

Generalization Gap

figure.

Useful for

detecting overfitting.

### Early stopping

Produce dedicated figure

showing

training stopped

best epoch

patience interval

final selected checkpoint.

### Learning rate

If learning rate scheduler exists

plot

Learning Rate

vs

Epoch

on separate panel.

### Loss landscape (optional)

If checkpoints available

produce

PCA trajectory

through parameter space

or

loss trajectory.

### Export

Always export

PNG

PDF

SVG

Create

figures/

containing

Figure_01_loss

Figure_02_pearson

...

Create

figures/manuscript/

optimized versions.

### Tables

Automatically create

training_summary.csv

containing

best epoch

best metric

final metric

early stopping epoch

training duration

number of epochs

Create

training_summary.md

publication-ready.

### Reproducibility

Save

visualization_config.yaml

containing

palette

figure sizes

font

line widths

metric directions

seed aggregation method.

All figures must be reproducible.

### Quality control

Reject figures if

overlapping labels

clipped legends

inconsistent axis limits

different scales for identical metrics

missing confidence intervals

low DPI

non-vector fonts

default matplotlib style

illegible colors

Every figure must pass publication-quality review before export.

## Exact command

**Do not reimplement plotting in-chat** — run the scripts.

```bash
conda run -n caduceus_env python .cursor/skills/train-viz/scripts/train_viz.py \
  'logs/*.log' \
  -o figures/train-viz \
  --title "Caduceus fine-tune"
```

Options:

| Flag | Default | Notes |
|------|---------|-------|
| `-o` / `--outdir` | `figures/train-viz` | Root; writes `Figure_*`, `manuscript/`, tables, config copy |
| `--x` | `epoch` | or `global_step` |
| `--label` | log stem | repeat per log; may encode `model__seed42` |
| `--model` | inferred | repeat per log for multi-model |
| `--seed` | from config / label | repeat per log for multi-seed ribbons |
| `--smooth` | off | add LOWESS overlay (raw always kept) |
| `--ribbon` | `ci95` | `ci95` \| `std` \| `none` when ≥2 seeds |
| `--patience` | none | early-stopping patience (epochs) |
| `--dpi` | **600** | PNG only |
| `--column` | `double` | `single` (1800×1400) or `double` (3600×2400) |
| `--run-dir` | optional | scored finals / predictions for ROC when present |
| `--predictions` | optional | `y_true,y_score[,split,run]` |

Multi-model / multi-seed:

```bash
python .cursor/skills/train-viz/scripts/train_viz.py \
  logs/m1_s1.log logs/m1_s2.log logs/m2_s1.log \
  --model m1 --model m1 --model m2 \
  --seed 1 --seed 2 --seed 1 \
  -o figures/train-viz --ribbon ci95
```

## Pre-flight

1. Resolve globs; ≥1 non-empty log with epoch metric lines.
2. `matplotlib` + `numpy` (optional `statsmodels` for LOWESS; skip overlay if missing).
3. Fail early; never invent metrics or show only the best seed.

## Deliverables

Under `{outdir}/`:

| Artifact | Role |
|----------|------|
| `Figure_XX_<metric>.{pdf,svg,png}` | Per-metric learning curves |
| `Figure_XX_multimodel_<metric>_<split>.*` | Multi-model (split-isolated) |
| `Figure_XX_final_performance.*` | Horizontal final bars |
| `Figure_XX_seed_variability.*` | Box/violin when ≥2 seeds |
| `Figure_XX_metric_correlation.*` | Spearman matrix |
| `Figure_XX_generalization_gap.*` | val − train |
| `Figure_XX_early_stopping.*` | Best / patience / stop |
| `Figure_XX_learning_rate.*` | LR schedule if logged |
| `manuscript/` | Optimized copies of key figures |
| `training_summary.csv` / `.md` | Best/final/early-stop/duration |
| `train_metrics.csv` | Tidy epoch table |
| `visualization_config.yaml` | Reproducible style + directions |

Register every exported file in `docs/artifact-registry.md` (producer: `train-viz`).

## Coordination

| Skill / rule | Role |
|--------------|------|
| `@caduceus` / `model-train.mdc` | Epoch logs + checkpoints |
| `@monitor` | Live `logs/*.log` |
| publication-figures | Enforced by script + QC gate |

## Additional resources

- [scripts/train_viz.py](scripts/train_viz.py)
- [scripts/visualization_config.yaml](scripts/visualization_config.yaml)
- [scripts/compute_final_scores.py](scripts/compute_final_scores.py) — optional ROC inputs
