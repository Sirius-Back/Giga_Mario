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

- **Default (implemented):** **union-find connected components** on the
  bipartite region↔k-mer contingency graph. Regions that share ≥ `min_shared`
  identical ACGT k-mers (default 1) are united into one component. Each
  connected component becomes one **fold** label; train/val/test are assigned
  **at fold (component) grain** via Caduceus-aligned ratios (`seed`), not per
  region independently.
- **Not used:** Louvain/Leiden modularity, spectral/Laplacian clustering, or
  Markov clustering (MCL). Those remain future options on the same graph.
- **ZSV:** IDs labeled zsv in `fold.csv` are held out before CC assignment.
- Must not require resolving the full repeat graph or all pairwise distances.

# Saved graph artifacts

Every pangenome split writes a reusable graph under `{outdir}/graph/`
(`save_graph=True` by default):

| File | Content |
|------|---------|
| `contingency_graph.npz` | `cluster_ids`, `edge_u`, `edge_v`, `edge_w` (int32) |
| `ids.txt` | Region IDs in array index order |
| `nodes.tsv` | `ID\|cluster` |
| `edges.tsv` | `source\|target\|weight` (capped co-occurrence edges) |
| `contingency_graph_meta.json` | k, min_shared, max_edges, clustering method note |

**Important:** `cluster_ids` come from the **full** streaming contingency
union-find. `edges.*` are a **capped** region–region edge list (≤ `max_edges`,
default 100 000) for visualization / figure rebuild — reloading CC from the
capped edge list alone may not recover every component merge.

Reload / replot without re-adapt::

  python -c "from src.splits.pangenome import load_contingency_graph, plot_pangenome_contingency_from_artifacts; ..."
  # or
  python -m src.runs.run17_pangenome_CDS_legnet.plot_contingency_graph

Diagnostics also under `figures/` (connected nodes only; fold + train_test panels).

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
    Diagnostics: `{outdir}/graph/` (npz + nodes/edges TSV — reusable) and
    `figures/contingency_graph.*` + `Figure_pangenome_contingency_fold_train_test.*`.
    Clustering = union-find CC (not modularity / Laplacian / MCL).

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
