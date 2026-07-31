---
id: mmseqs
name: MMseqs2 homology split (SBS)
aliases:
  - MMseqs2
  - mmseqs2
  - mmseqs_sbs
  - sequence_identity
---

# Description

Sequence-identity-aware train / validation / test assignment via
[MMseqs2](https://github.com/soedinglab/MMseqs2) inside the Split-by-similarity
(SBS) family. Primary sequence input is the panel **`MARKED/`** directory (one
FASTA per region ID). MMseqs2 **clusters** homologous regions
(`easy-cluster`); each cluster is a fold. Entire folds are assigned to
train / validation / test so homologous groups do not leak across splits.

Unlike `gc` / `kmer` (composition features) and `hashfrag` (BLAST+hashFrag
orthogonal splits), this strategy uses **MMseqs2 identity clustering**. Dense
all-vs-all distance matrices are **not** used for production clustering
(\(O(n^2)\)); keep `MMseqsDistanceBackend` for small-n / legacy only.

Output matches the universal contract: `split.csv` with `ID|train_test|fold`,
then `/split` materializes `SPLIT/`. Default downstream model: **LegNet**.

# Split

train:
- All regions belonging to folds assigned to train (fold-grain;
  **locked ratios train:val:test = 60:20:20**).

validation:
- All regions belonging to folds assigned to validation (20%).

test:
- All regions belonging to folds assigned to test (20%).

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; never
  clustered into train/val/test).

# Inputs

| Input | Required | Notes |
|-------|----------|-------|
| **MARKED** / **fna** | yes | Panel MARKED dir (preferred) or multi-FASTA |
| **fold.csv** | optional | ZSV holdout + fold filter before clustering |
| **stratification.csv** | optional | Fold-grain stratification (SBS in-built) |
| **ID.csv** | optional | Genome labels for PCA diagnostics |
| **mmseqs** | yes | Binary on PATH or explicit path |
| **min_seq_id** | yes (choose + record) | MMseqs `--min-seq-id` for `easy-cluster` |
| **sensitivity (`-s`)** | optional | Default document in method-decision when set |

Do **not** feed LegNet-stitched `PARSED` adapter sequences by default.

# Pipeline

1. **load** — MARKED / FNA → region sequences (`src.splits.sbs.fna_io`).
2. **fold filter** — hold out ZSV from `fold.csv`.
3. **mmseqs easy-cluster** — cluster-first identity clustering (not dense
   all-vs-all for production panels).
4. **SBS assign** — cluster = fold; fold→train/test/val at
   `ratios=(0.6, 0.2, 0.2)` (train:test:val); optional stratification;
   PCA diagnostics.
5. **materialize** — `/split` → `SPLIT/`.
6. **LegNet** — `/train train=legnet` on the materialized trees.

# Implementations

- name: GigaMario MMseqs2 SBS split
  url: https://github.com/soedinglab/MMseqs2
  paper: Steinegger & Söding, Nature Biotechnology 2017 (MMseqs2)
  split_location: `src/splits/mmseqs.py` + `src/splits/sbs/backends/mmseqs.py`
  run: |
    python -m src.pipeline.split_predict \
      --outdir output/mmseqs_split \
      --type mmseqs \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --marked ready_legnet/MARKED \
      --seed 42 \
      --ratios 0.6,0.2,0.2 \
      --plot
    # then materialize + LegNet:
    # /split  → SPLIT/
    # /train train=legnet folders=<SPLIT>
  notes: |
    Locked ratios: train/val/test = 60:20:20 → CLI/API train:test:val
    `0.6,0.2,0.2` (same masses).
    Production: `mmseqs easy-cluster` → cluster TSV → folds.
    Legacy only: `MMseqsDistanceBackend` easy-search distance matrix (small-n).
    Requires `mmseqs` on PATH. Record `--min-seq-id` and `-s` in method-decision.
    `/split-generate` reads this caption to emit `src/splits/mmseqs.py`;
    `/split` runs split-predict + materialize; then `/train` LegNet.

# References

- MMseqs2: https://github.com/soedinglab/MMseqs2
- SBS architecture: `wiki/sbs.md`
- Pipeline contracts: `wiki/architecture.md`
- Peer strategies: `splits/hashfrag.md`, `splits/gc.md`, `splits/kmer.md`
- Skill: `.cursor/skills/mmseqs/SKILL.md`
- Split generate: `.cursor/skills/split-generate/SKILL.md`
- Train / LegNet: `.cursor/skills/train/SKILL.md`
