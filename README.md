# GigaMario

Toolkit for preparing genomic intervals, linking prediction targets, building leakage-aware train/val/test partitions, and training DNA foundation models (Caduceus, LegNet, …).

## What it does

At a high level the tool walks raw genomes and annotations through a fixed stage graph:

1. **Adapt** — mark genomic intervals from annotations + FASTA (task-dependent windows such as gene flanks or promoters) and record interval intersections.
2. **parse_data** — turn marked sequences into a model-ready parsed form under stable IDs.
3. **parse_target** — turn raw target tables (expression, assays, …) into a prediction table aligned to those IDs.
4. **split-predict** — assign each ID to train / test / val (and optional folds), optionally using stratification, pre-made folds, or intersection-aware strategies.
5. **split** — materialize partitioned sequence and prediction folders from that assignment table.
6. **train** / **train-viz** — fine-tune a chosen model on the materialized splits and plot training curves.

Helpers (`id_gen`, `id_rule`) build and filter the shared ID table used by folds and stratification. An **adversarial** stage can rebuild a same-shaped panel for stress tests.

```text
GTF + FNA ──adapt──► marked sequences + intersections
                         │
                         └──parse_data──► parsed sequences
raw targets ──parse_target──► predictions (per ID)
                         │
marked / annotations / optional strat·fold ──split-predict──► assignment table
                         │
parsed sequences + predictions + assignment ──split──► partitioned folders
                         │
                         └──train──► logs ──train-viz──► figures
```

Exact on-disk layouts and column schemas live in [wiki/architecture.md](wiki/architecture.md). Migration from today’s Caduceus/LegNet scripts is tracked in [refactoring.md](refactoring.md).

## Install (conda-preferred)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
conda env update -f environment.yml --prune
```

Editable install:

```bash
python -m pip install -e .
```

Verify:

```bash
python -c "import GigaMario; print(GigaMario.__version__)"
```

Model-specific envs (`caduceus_env`, `legnet`) are used for training; see the wiki and skill docs for stage commands while the public CLI is still landing.
