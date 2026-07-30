---
id: pangenome
name: Pangenome graph split
aliases:
  - cactus
  - cactus_split
  - pangenome
---

# Description

Construct a pangenome-style **repeat / contingency graph** from
**pangenome-window** sequences and assign train / validation / test at
**connected-component** grain. Highly similar regions that share k-mers fall
into the same fold without pairwise distance matrices or full bubble resolution.

**Important:** the pangenome genomic window may differ from panel `MARKED`
(LegNet/Caduceus). Do **not** silently reuse panel `MARKED`. Produce
`MARKED_pangenome` from raw via adapt (`@preprocess` / `src.pipeline.adapt`) —
documented **A2A** handoff — then filter to `MARKED_parsed`.

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

1. **Adapt (A2A → `@preprocess` / `adapt`)** — `raw` (GTF+FNA+ID.csv) →
   `MARKED_pangenome` with the **pangenome** `environment`/`window` (may differ
   from panel MARKED). Writes `intersect_pangenome.csv` when adapting here.
2. **Filter** — `MARKED_pangenome` ∩ `PARSED` → `MARKED_parsed`
   (`python -m src.splits.intersect_pangenome`).
3. **Repeat / contingency graph** (C++) on `MARKED_parsed`: stream k-mers;
   union regions that share ≥1 k-mer (no all-pairs distances).
4. **Cluster**: connected components of the region contingency graph.
5. **Assign** clusters → train / val / test (+ optional ZSV).
6. **Render**: region co-occurrence graph with **connected nodes only**.

Opt-in only: `reuse_panel_marked=True` when the panel MARKED window is
**intentionally identical** to the pangenome window (smoke tests).

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
  split_location: `src/splits/pangenome.py` + `src/splits/pangenome_native/` + `src/splits/intersect_pangenome.py`
  run: |
    # Preferred: adapt pangenome window from raw, then split-predict
    python -m src.splits.pangenome_native.build
    python -m src.pipeline.split_predict \
      --outdir output/pangenome_split \
      --type pangenome \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --parsed ready_legnet/PARSED \
      --gtf-dir raw/gtf \
      --fna-dir raw/fna \
      --environment gene \
      --window '{"pos1":-100,"pos2":100}' \
      --seed 42 \
      --kmer-size 21 \
      --plot

    # Or pass an existing MARKED_pangenome tree:
    #   --marked-pangenome path/to/MARKED_pangenome
  notes: |
    A2A: if MARKED_pangenome is missing, agents must invoke @preprocess/adapt
    with the pangenome window — do not assume ready_*/MARKED matches.
    Filter: MARKED_pangenome ∩ PARSED → MARKED_parsed.
    Diagnostics: figures/contingency_graph.{json,dot,pdf,png}.

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
- Adapt stage: `wiki/architecture.md` (`src.pipeline.adapt`)
