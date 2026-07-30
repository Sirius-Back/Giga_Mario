---
id: pangenome
name: Pangenome graph split
aliases:
  - cactus
  - cactus_split
  - pangenome
---

# Description

Construct a pangenome-style **repeat / contingency graph** from filtered MARKED
sequences and assign train / validation / test at **connected-component** grain.
Highly similar regions that share k-mers fall into the same fold without building
pairwise distance matrices or fully resolving graph bubbles.

Inspired by Minigraph-Cactus-style pangenome graphs, but scoped to ML dataset
splitting. Fast path: C++ rolling k-mer contingency + union-find
(`src/splits/pangenome_native`).

# Split

train:
- All regions in contingency clusters assigned to train (fold-grain;
  Caduceus-aligned ratios by default).

validation:
- All regions in contingency clusters assigned to validation.

test:
- All regions in contingency clusters assigned to test.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; never
  clustered into train/val/test).

# Pipeline

1. **Adapt** (external): `raw` → panel `MARKED` (+ PARSED / PREDICT). Optional
   materialization of filtered `MARKED_pangenome` under the split outdir.
2. **Filter / intersect**: retain only IDs present in both MARKED and PARSED
   (`intersect_pangenome`).
3. **Repeat / contingency graph** (C++): stream k-mers; collapse identical k-mer
   keys; union regions that share ≥1 k-mer (no all-pairs distances).
4. **Cluster**: connected components of the region contingency graph.
5. **Assign** clusters → train / val / test (+ optional ZSV).
6. **Render**: region co-occurrence graph with **connected nodes only**
   (JSON, DOT, PDF/PNG).

# Graph construction

- Extract overlapping ACGT k-mers (default `k=21`; rolling 2-bit codes).
- Reuse identical k-mer keys as shared graph nodes (contingency join).
- Single streaming pass; scales approximately linearly with input bases.
- Do **not** compute dense pairwise sequence distances.

# Clustering

- Default: **connected components** via union-find on shared-k-mer contingency.
- Alternatives (future): Leiden / Louvain / label propagation on the same graph.
- Must not require resolving the full repeat graph or all pairwise distances.

# Implementations

- name: GigaMario pangenome contingency split
  url: https://github.com/ (local toolkit)
  paper: —
  split_location: `src/splits/pangenome.py` + `src/splits/pangenome_native/`
  run: |
    python -m src.splits.pangenome_native.build
    python -m src.pipeline.split_predict \
      --outdir output/pangenome_split \
      --type pangenome \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --marked ready_legnet/MARKED \
      --seed 42 \
      --kmer-size 21 \
      --plot
  notes: |
    Filter: MARKED ∩ PARSED (PARSED defaults to `<marked>/../PARSED`).
    Optional genome subset via strategy API `genomes=[...]`.
    Diagnostics: `figures/contingency_graph.{json,dot,pdf,png}`.

- name: Minigraph-Cactus
  url: https://github.com/ComparativeGenomicsToolkit/cactus
  paper: Minigraph-Cactus Pangenome Pipeline
  split_location: Graph construction / chromosome graph partitioning
  run:
  notes: |
    Reference pangenome construction workflow. Does not perform train/test
    splitting; inspires graph-building ideas used here.

# References

- Minigraph-Cactus (Comparative Genomics Toolkit)
- Project SBS / split-generate contracts: `wiki/sbs.md`, `wiki/split-generate.md`
