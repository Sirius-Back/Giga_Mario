# src/

Canonical project code for the Caduceus pipeline and future Python modules.

## Layout

| Path | Role |
|------|------|
| `pipeline/` | Universal stages (`id_gen`…`adversarial`); `@adapt` / `@prepare-target` |
| `preprocessing.py` | `@adapt-legacy` — `raw/` → `data_ready/` |
| `splits/` | `@split` — `python -m src.splits.main` |
| `caduceus.py` | `@caduceus` — `python -m src.caduceus` |
| `metrics_logging.py` | `metrics.md` TorchMetrics helpers |
| `train_viz/` | `@train-viz` — `python -m src.train_viz` |
| `runs/` | `@caduceus-full` — `python -m src.runs.caduceus_full` |
| `ready_analysis.py` | `@analyze-ready-data` — barplots + GC/length densities |
| `summarize_geo.py` | `@summarize_GEO` — mean-merge GEO TPM per assembly |
| `get_mpra.py` | Wide TPM → LegNet soft-classification bin fractions (`python -m src.get_mpra`) |
| `sbatch/` | SLURM wrappers that call `src/` modules |
| `_archive/` | Superseded legacy scripts (audit only) |

## Entry points

```bash
python -m pytest tests/pipeline -q
python -m src.pipeline.id_gen --gtf tests/fixtures/mini_raw/gtf --outdir /tmp/ids
python -m src.pipeline.adapt --gtf … --fna … --id-csv … --outdir …
python src/preprocessing.py --raw raw --out data_ready
python -m src.splits.main --strategy random --raw raw --ready ready
python -m src.caduceus --splits-dir splits/random/M1
python -m src.train_viz --models runs/caduceus/M1 -o figures/train-viz/M1
python -m src.ready_analysis --ready-dir ready --outdir output/ready_analysis
python src/summarize_geo.py --mappings prokaryotes/expr_file_mappings.csv
python -m src.get_mpra --tpm prokaryotes/tpm --outfolder prokaryotes/mpra
python -m src.runs.caduceus_full --strategy random --ready ready --out-root output/random
```

## SLURM

```bash
sbatch src/sbatch/preprocess_raw.sbatch
sbatch src/sbatch/split_random.sbatch
sbatch src/sbatch/caduceus_train.sbatch
sbatch src/sbatch/caduceus_full_random.sbatch
```

New pipeline code belongs under `src/` (not a top-level `scripts/` directory).
