---
id: gc
name: GC composition split (SBS)
aliases:
  - GC
  - delta_gc
  - gc_sbs
  - gc_aaa
---

# Description

Split-by-similarity (SBS) strategy that embeds each region in a small **feature space** — currently **GC (%)** and **AAA (%)** — then clusters those features into folds. Entire folds are assigned to train / validation / test to reduce composition leakage relative to a fully random per-region split. Zero-shot IDs from `fold.csv` are held out and never clustered.

Dense pairwise distance matrices are **not** used for clustering (\(O(n^2)\)). Future backends may add more features or MMseqs-derived embeddings while reusing the same SBS contracts.

# Split

train:
- All regions belonging to folds assigned to train (fold-grain; Caduceus-aligned ratios by default).

validation:
- All regions belonging to folds assigned to validation.

test:
- All regions belonging to folds assigned to test.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; not used in clustering or train/val/test).

# Implementations

- name: GigaMario SBS / GC+AAA features
  url: https://github.com/ (local toolkit)
  paper: —
  split_location: `src/splits/gc.py` + `src/splits/sbs/`
  run: |
    python -m src.pipeline.split_predict \
      --outdir output/gc_split \
      --type gc \
      --id-csv ready_caduceus/ID.csv \
      --fold ready_caduceus/fold.csv \
      --marked ready_caduceus/MARKED \
      --seed 42 \
      --cluster-method dbscan \
      --plot \
      --custom-label-column strat1 \
      --stratification path/to/stratification.csv
  notes: |
    Features: `GC_pct`, `AAA_pct` via `GcAaaFeatureBackend`.
    Clustering default: DBSCAN (auto eps). Also: kmeans, kmeans_elbow,
    hierarchical, pca_kmeans, auto (DBSCAN→elbow fallback).
    Stratification: aggregate per fold then fold→train/val/test.
    Diagnostics: PCA of the feature table colored by cluster, train/test/val,
    genome (ID.csv), and optional custom stratification column — cnsplots + Altair.

# References

- SBS architecture: `wiki/sbs.md`
- Pipeline contracts: `wiki/architecture.md`
- Random baseline ratios: `splits/random.md`
