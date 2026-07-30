---
id: blastp
name: BLASTP homology split (SBS)
aliases:
  - BLASTP
  - protein_homology
  - blastp_sbs
---

# Description

Protein-homology-aware train / validation / test assignment via **BLASTP**
inside the Split-by-similarity (SBS) family. Unlike `hashfrag` (DNA `blastn` on
panel `MARKED/`) and `gc`/`kmer` (composition features), this strategy starts
from **raw genomes** (`fna` + `gtf`), adapts windows, filters to panel
`PARSED` IDs, translates under a genetic code, then clusters on BLASTP-derived
similarity — **not** a dense all-vs-all distance matrix when a candidate
heuristic is available.

Output matches the universal contract: `split.csv` with `ID|train_test|fold`,
then `/split` materializes `SPLIT/`.

# Split

train:
- All regions belonging to folds assigned to train (fold-grain;
  Caduceus-aligned ratios by default).

validation:
- All regions belonging to folds assigned to validation.

test:
- All regions belonging to folds assigned to test.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; never
  clustered into train/val/test).

# Inputs

| Input | Required | Notes |
|-------|----------|-------|
| **fna** | yes | Genome FASTA dir (raw), **instead of** feeding panel `MARKED/` |
| **gtf** | yes | Annotation dir (raw), with `fna` |
| **window** | yes | Adapt window (e.g. gene ± flanking) |
| **genetic code** | yes (default **universal**) | Translation table for DNA → protein before BLASTP |
| **PARSED** | yes (filter step) | Keep only IDs present in panel `PARSED/` |
| **fold.csv** | optional | ZSV holdout + fold filter before BLASTP |
| **stratification.csv** | optional | Fold-grain stratification (SBS in-built) |

Do **not** silently reuse panel `MARKED` unless the blastp window is
intentionally identical (opt-in only).

# Pipeline

1. **adapt** — `raw` (`fna` + `gtf`) → `MARKED` + `intersect.csv`
   (`src.pipeline.adapt` / `@preprocess`).
2. **filter** (to implement) — keep only those IDs that exist in `PARSED`
   → filtered MARKED subset for BLASTP.
3. **sbs:**
   1. filter fold (ZSV / fold.csv holdouts out of clustering)
   2. run **blastp** — may implement a **heuristic here** to avoid full
      all-vs-all
   3. cluster, QC / control, stratify (SBS in-built:
      `assign_from_features` / fold→train/val/test + optional strat)

# Implementations

- name: GigaMario BLASTP SBS split
  url: https://github.com/ (local toolkit)
  paper: —
  split_location: `src/splits/blastp.py` (+ filter helper; SBS assign/viz)
  run: |
    # Preferred: adapt from raw, filter to PARSED, then split-predict
    python -m src.pipeline.split_predict \
      --outdir output/blastp_split \
      --type blastp \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --parsed ready_legnet/PARSED \
      --gtf-dir raw/gtf \
      --fna-dir raw/fna \
      --environment gene \
      --window '{"pos1":-100,"pos2":100}' \
      --genetic-code universal \
      --seed 42 \
      --plot
  notes: |
    Genetic code default: `universal`.
    Step (2) filter: IDs ∩ PARSED only (must be written; mirror
    `intersect_pangenome` pattern — do not invent labels).
    Step (3.2): BLASTP with optional candidate heuristic (no dense n×n
    requirement for clustering).
    Step (3.3): reuse SBS clustering, control, and stratification.
    Requires BLAST+ (`blastp`, `makeblastdb`) on PATH when implemented.
    `/split-generate` reads this caption to emit `src/splits/blastp.py`;
    `/split` then runs split-predict + materialize.

# References

- SBS architecture: `wiki/sbs.md`
- Pipeline contracts: `wiki/architecture.md`
- Peer strategies: `splits/pangenome.md`, `splits/hashfrag.md`, `splits/gc.md`
- Skill: `.cursor/skills/blastp/SKILL.md`
- Split generate: `.cursor/skills/split-generate/SKILL.md`
