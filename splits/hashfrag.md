---
id: hashfrag
name: hashFrag homology-aware split
aliases:
  - hashFrag
  - homology
  - orthogonal_homology
---

# Description

Homology-aware train/test(/val) assignment via the external
[hashFrag](https://github.com/de-Boer-Lab/hashFrag) CLI (BLAST pairwise scores →
candidate filter → homologous groups → orthogonal splits). Primary sequence
input is the panel **`MARKED/`** directory (one FASTA per region ID). Reverse
complements are left to hashFrag (do not invent `_Reversed` mates upstream).

Unlike `gc` (composition features + SBS clustering), this strategy uses
**alignment-score homology**. Output still matches the universal contract:
`split.csv` with `ID|train_test|fold`, then `/split` materializes `SPLIT/`.

# Split

train:
- Regions whose homologous group is assigned to the hashFrag train pool, minus
  a Caduceus-aligned validation carve-out from that pool.

validation:
- Carved from the hashFrag train pool (default: 10% of the train pool) so val
  stays homology-orthogonal to test. hashFrag itself emits only train/test.

test:
- Regions whose homologous group is assigned to the hashFrag test split.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; never
  written into the hashFrag FASTA / BLAST DB).

# Implementations

- name: GigaMario hashFrag orthogonal splits
  url: https://github.com/de-Boer-Lab/hashFrag
  paper: https://www.biorxiv.org/content/10.1101/2025.01.22.634321v2
  split_location: `src/splits/hashfrag.py` + `src.pipeline.split_predict`
  run: |
    python -m src.pipeline.split_predict \
      --outdir output/hashfrag_split \
      --type hashfrag \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --marked ready_legnet/MARKED \
      --threshold 60 \
      --seed 42 \
      --threads 16
  notes: |
    Requires `hashFrag` and BLAST+ (`blastn`, `makeblastdb`) on PATH.
    Threshold `-t` / `--threshold` is obligatory (no default invent).
    FASTA headers use `hf_{ID}` tokens so numeric region IDs stay strings under
    BLAST/pandas (hashFrag union-find bug on int IDs).
    `fold` column stores homologous-group id from hashFrag when available.
    Optional `--p-train` / `--p-test` (must sum to 1); val carved afterward.
    Large panels: prefer SLURM / `blastn_array_module` (see hashfrag skill).

# References

- hashFrag docs: https://hashfrag.readthedocs.io/en/latest/
- Skill: `.cursor/skills/hashfrag/SKILL.md`
- Pipeline contracts: `wiki/architecture.md`
- Peer strategies: `splits/random.md`, `splits/gc.md`
