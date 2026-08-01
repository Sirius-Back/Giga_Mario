---
id: pangenome
name: Pangenome graph split
aliases:
  - cactus
  - cactus_split
  - pangenome
---

# Description

Construct a pangenome-style **repeat hash graph** from **pangenome-window**
sequences. Graph **nodes are k-mer hashes**; each sequence links to many
hashes. Hash nodes are clustered (union-find on repeat co-occurrence), then
each sequence gets a **majority** hash-cluster as its fold. Train / validation
/ test are assigned at **fold grain**.

**Important:** the pangenome genomic window may differ from panel `MARKED`
(LegNet/Caduceus). Do **not** silently reuse panel `MARKED`. Produce
`MARKED_pangenome` from raw via adapt (`@preprocess` / `src.pipeline.adapt`) —
documented **A2A** handoff — then filter to `MARKED_parsed`.

# Split

train:
- All regions whose majority hash-cluster fold is assigned to train
  (Caduceus-aligned ratios by default).

validation:
- All regions whose fold is assigned to validation.

test:
- All regions whose fold is assigned to test.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out; never
  clustered into train/val/test).

# Pipeline

1. **Adapt (A2A → `@preprocess` / `adapt`)** — `raw` (GTF+FNA+ID.csv) →
   `MARKED_pangenome` with the **pangenome** `environment`/`window` (may differ
   from panel MARKED). Writes `intersect_pangenome.csv` when adapting here.
2. **Filter** — `MARKED_pangenome` ∩ `PARSED` → `MARKED_parsed`
   (`python -m src.splits.intersect_pangenome`).
3. **Hash graph** (C++) on `MARKED_parsed`: extract ACGT k-mer hashes; keep
   **repeat** hashes with document frequency ≥ `min_df` (default 2).
4. **Cluster hashes**: union-find on hash nodes — unite hash pairs that
   co-occur in ≥2 sequences (a single multi-motif bridge does not merge
   families).
5. **Sequence → fold**: majority hash-cluster per sequence (ties → smaller id);
   sequences without repeat hashes get a singleton fold.
6. **Assign** folds → train / val / test (+ optional ZSV).
7. **Render**: capped region–region co-occurrence edges (connected nodes only).

Opt-in only: `reuse_panel_marked=True` when the panel MARKED window is
**intentionally identical** to the pangenome window (smoke tests).

# Graph construction

- Extract overlapping ACGT k-mers (default `k=21`; rolling 2-bit codes /
  `uint64` hashes).
- One sequence ↔ many hash nodes; link stored during the streaming pass.
- Repeat filter: hash kept only if it appears in ≥ `min_df` sequences.
- Do **not** compute dense pairwise sequence distances.
- Do **not** build a full clique on all regions.

# Clustering

- **Default (implemented):** `hash_majority` —
  1. Nodes = k-mer hashes with `df ≥ min_df`.
  2. Union-find CC on hash nodes (unite pairs co-occurring in ≥2 sequences).
  3. Each sequence fold = **majority** of its repeat-hash cluster ids.
  4. Train/val/test assigned **at fold grain** via Caduceus-aligned ratios
     (`seed`), not per region independently.
- **Legacy (opt-in):** `region_contingency` — union-find directly on regions
  that share ≥1 k-mer (first-owner join). Use
  `--pangenome-cluster-method region_contingency`.
- **Not used as default:** Louvain/Leiden modularity, spectral/Laplacian, MCL
  (optional Louvain refine remains available for oversized folds).
- **ZSV:** IDs labeled zsv in `fold.csv` are held out before fold→split
  assignment.

# Saved graph artifacts

Every pangenome split writes a reusable graph under `{outdir}/graph/`
(`save_graph=True` by default):

| File | Content |
|------|---------|
| `contingency_graph.npz` | `cluster_ids` (per-sequence folds), `edge_u/v/w` (int32) |
| `ids.txt` | Region IDs in array index order |
| `nodes.tsv` | `ID\|cluster` |
| `edges.tsv` | `source\|target\|weight` (capped region–region shared-repeat edges) |
| `contingency_graph_meta.json` | k, min_df, method=`hash_majority`, notes |

**Important:** `cluster_ids` are **majority folds** from the hash UF. `edges.*`
are a **capped** region–region list for visualization — not the full hash
graph.

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
