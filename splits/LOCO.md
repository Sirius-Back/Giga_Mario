---
id: loco
name: Leave-one-chromosome-out (chromosome-grain) split
aliases:
  - LOCO
  - loco
  - chromosome
  - chrom_split
  - leave_one_chromosome
---

# Description

Chromosome-aware train / validation / test assignment. The assignment grain is
**(organism, chromosome)**: every panel region (gene / window) from the **same
genome** on the **same chromosome** receives the **same** `train_test` label.
Regions are never split across folds within one chromosome of one organism.

Chromosome folds are then mapped to train / validation / test with
**stratification by chromosome number** (normalized ordinal / label such as
`1`, `2`, …, `X`, `Y`, `MT` across genomes). The intent is to keep homologous
chromosome ranks balanced across roles (e.g. not dump every species' chr1 into
test) while still evaluating generalization to held-out chromosomes.

Unlike `random` (per-region), `gc` / `kmer` / `mmseqs` (similarity clusters),
and `hashfrag` / `blastp` / `pangenome` (homology or repeat graphs), this
strategy uses **only genomic coordinates** from `ID.csv` (`genome`, `chr`).
Dense distances and sequence clustering are **not** used.

Output matches the universal contract: `split.csv` with `ID|train_test|fold`,
then `/split` materializes `SPLIT/`.

# Split

train:
- All regions whose `(genome, chr)` fold is assigned to train (chromosome
  grain; Caduceus-aligned ratios by default unless locked otherwise).

validation:
- All regions whose `(genome, chr)` fold is assigned to validation.

test:
- All regions whose `(genome, chr)` fold is assigned to test.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; never
  enter chromosome→train/val/test assignment).

# Inputs

| Input | Required | Notes |
|-------|----------|-------|
| **ID.csv** | yes | Must provide `genome`, `chr`, and region `ID` (panel schema) |
| **fold.csv** | optional | ZSV holdout before chrom assignment |
| **seed** | yes (default 42) | Fold→train/val/test shuffle within chromosome-number strata |
| **ratios** | optional | Default Caduceus-aligned train:test:val unless overridden |

Do **not** invent chromosome labels from sequence. Use the `chr` field already
present in panel `ID.csv` (accessions or names as stored after adapt).

# Chromosome number (stratification key)

1. Read `chr` per region from `ID.csv`.
2. Map each contig accession / name to a **chromosome number token**
   (e.g. RefSeq `NC_000001.11` → `1`, `NC_000023.11` → `X`, mitochondrial →
   `MT`). Unplaced / random / alt contigs get a dedicated token (e.g.
   `unplaced`) and form their own stratum — do not silently merge them into
   numbered autosomes.
3. Fold id = stable string `"{genome}|{chr}"` (raw contig id, not only the
   number token) so two species' chr1 remain **distinct folds**.
4. Stratification key for fold→train/val/test = the **chromosome number token**
   from step 2 (shared across genomes when the token matches).

# Pipeline

1. **load** — panel `ID.csv` → region rows with `genome`, `chr`, `ID`.
2. **ZSV filter** — drop `fold.csv` zero-shot IDs.
3. **fold** — group remaining IDs by `(genome, chr)`; each group is one fold.
4. **stratify** — assign folds to train / val / test at fold grain, stratified
   by chromosome number token (step 2 above); seeded RNG within strata;
   Caduceus-aligned ratios by default.
5. **write** — `split.csv` (`ID|train_test|fold`); `fold` stores
   `genome|chr`.
6. **materialize** — `/split` → `SPLIT/`.

# Implementations

- name: GigaMario LOCO / chromosome-grain split
  url: https://github.com/ (local toolkit)
  paper: —
  split_location: `src/splits/loco.py` (+ thin wiring in `src.pipeline.split_predict`)
  run: |
    python -m src.pipeline.split_predict \
      --outdir output/loco_split \
      --type loco \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --seed 42 \
      --plot
  notes: |
    No MARKED / BLAST / MMseqs required — metadata-only from ID.csv.
    Fold grain is strictly (genome, chr); never split one chrom of one
    organism across train/val/test.
    Stratification is by normalized chromosome number across organisms.
    `/split-generate` may emit `src/splits/loco.py` from this caption;
    `/split` runs split-predict + materialize.

# References

- Peer strategies: `splits/random.md`, `splits/gc.md`, `splits/mmseqs.md`
- Panel ID schema: `genome|chr|pos1|pos2|…|ID` (`ready_*/ID.csv`)
- Pipeline contracts: `wiki/architecture.md`, `wiki/split.md`
- Split generate: `.cursor/skills/split-generate/SKILL.md`
