---
id: paralogs_only
name: Orthogroup-representative train / paralog remainder split
aliases:
  - paralogs_only
  - orthogroup_rep_train
  - homology_rep_split
---

# Description

Homology-aware train / validation / test assignment from the **Ensembl Compara
ortholog–paralog graph** (mammals-11 panel; see `mag/homology_graph/`).

A **new C++ native** walks the graph, partitions nodes into **orthologous
groups** (connected components on **ortholog** edges only), and keeps
**exactly one representative gene per orthogroup**. That representative set
becomes **train**. Every other mapped panel gene (other co-orthologs and
paralogs left out of the representative set, plus optional unmapped policy)
is randomly split **½ / ½** into **test** and **validation**.

Intent: train on a non-redundant ortholog-reduced set; evaluate on leftover
copies that are largely co-orthologs / paralogs — hence the id
`paralogs_only`. Dense pairwise sequence distances and MMseqs/BLAST clustering
are **not** used.

This is **not** the FASTA orthoparagroup extractor in
`mag/src/orthoparagroups/` (that writes multi-sequence `.fna` clusters). It
**reuses** the same edge/node schema and ortholog-CC idea, but emits
`split.csv` (`ID|train_test|fold`) for `/split`.

# Split

train:
- Exactly **one** panel region per orthogroup (the selected representative).
- Size is determined by the number of orthogroups with ≥1 mappable panel
  gene — **not** Caduceus ~81/9/10 ratios.

validation:
- Half of the **remainder** (all eligible panel IDs that are not train
  representatives), seeded random 50/50 with test.

test:
- The other half of the remainder.

zero_shot:
- IDs labeled `zsv` / `zeroshotvalidation` in `fold.csv` (held out before
  representative selection and remainder shuffle; never enter train/val/test).

# Inputs

| Input | Required | Notes |
|-------|----------|-------|
| **homology edges** | yes | `gene1\|genome1\|gene2\|genome2\|relation` with `relation ∈ {ortholog, paralog}` (default: `mag/homology_graph/edges.tsv.gz`) |
| **node ↔ panel map** | yes | Ensembl gene → panel `marked_id` / region `ID` (default: `mag/homology_graph/maps/nodes_extract.tsv` or enriched map filtered to non-empty `marked_id`) |
| **ID.csv** / panel IDs | yes | Universe of region IDs for `split.csv` |
| **fold.csv** | optional | ZSV holdout |
| **seed** | yes (default 42) | Representative tie-breaks + remainder shuffle |

Do **not** invent orthology from sequence alone. Rebuild the edge table only
via the documented homology-graph runner when Compara inputs change.

# Graph definitions

- **Node:** Ensembl gene stable ID × production species (`genome`).
- **Ortholog edge:** Compara `homology_type` containing `ortholog`
  (collapsed to relation=`ortholog` in the edge table).
- **Paralog edge:** `within_species_paralog` / `other_paralog` (relation=`paralog`).
- **Orthogroup (OG):** connected component of the subgraph induced by
  **ortholog edges only** (union-find / CC). Paralog edges do **not** merge
  orthogroups; they only explain why leftover nodes are enrichment for
  within-family copies.
- **Full homology component** (optional diagnostics): CC on ortholog∪paralog
  edges — same as `mag/homology_graph` components; not the train grain.

# Algorithm (C++ native — required for `/split-generate`)

Implement under `src/splits/paralogs_only_native/` (mirror
`src/splits/pangenome_native/` packaging). Prefer reusing graph load / adj /
`connected_components(ortholog_only)` patterns from
`mag/src/orthoparagroups/{graph,io}.cpp` rather than a third parallel graph
stack. Thin Python wrapper: `src/splits/paralogs_only.py` →
`split_predict type=paralogs_only`.

```
inputs: edges.tsv[.gz], nodes map (ensembl → marked_id), panel ID list, seed
        optional fold.csv (ZSV)

1. Load undirected edges; build ortho_adj and para_adj; drop self-loops.
2. Restrict to nodes with non-empty marked_id that appear in the panel ID
   universe (after ZSV removal).
3. Orthogroups ← connected_components(ortholog_only=true).
4. For each orthogroup G with at least one eligible node:
     pick exactly one representative r ∈ G
       default rule (Tentative): argmax paralog_degree(r);
       ties broken by seeded RNG, then stable gene id.
     Map r → panel region ID → label train.
     Record fold = orthogroup_id (string/int) on that ID.
5. Remainder R ← eligible panel IDs not in the representative set
     (other OG members + any unmapped_policy targets; see below).
6. Shuffle R with seed; assign first ⌈|R|/2⌉ → test, rest → validation
     (or exact 50/50 with last odd ID going to test — document in code).
     fold for remainder IDs = their orthogroup_id if mapped, else "unmapped".
7. Write split.csv: ID|train_test|fold
8. Optional artifacts under {outdir}/graph/:
     orthogroup_ids, representative table, counts per species / relation.
```

**Representative rule (Tentative):** max `paralog_degree`, seeded tie-break —
aligned with `mag/src/orthoparagroups/extract.cpp` species-rep heuristic, but
**one gene per OG total** (not one per species). Alternatives: uniform random
among OG members; prefer a locked reference species (e.g. `homo_sapiens`) when
present.

**Unmapped panel IDs** (no Ensembl edge endpoint / empty `marked_id`):
**Locked** — place into **remainder** (test/val pool) only; **never train**.

# Ratios (Locked by design)

| Role | Rule |
|------|------|
| train | \|representatives\| (one per OG) |
| validation | ≈ ½ of remainder |
| test | ≈ ½ of remainder |

Do **not** call `train_test_val_weights` Caduceus defaults for the primary
assignment. Remainder split weights are **locked** `test:val = 1:1`
(implementation may still use a tiny helper with `ratios=(0.5, 0.5)` over
remainder only).

# Pipeline

1. **Prereq** — mammals-11 Compara edge table + node map
   (`python -m src.run.homology_graph.build_mammals11` when rebuilding).
2. **ZSV filter** — drop `fold.csv` zero-shot IDs.
3. **C++ paralogs_only** — OG CC → 1-rep train → remainder 50/50 test/val.
4. **split.csv** → `/split` materialize `SPLIT/`.
5. **Diagnostics** (optional): counts of OG sizes, train size, fraction
   remainder that have ≥1 paralog edge, species composition of train vs
   remainder.

# Implementations

- name: GigaMario orthogroup-representative / paralog-remainder split
  url: https://github.com/ (local toolkit)
  paper: —
  split_location: |
    `src/splits/paralogs_only.py` + `src/splits/paralogs_only_native/`
    (graph schema shared with `mag/homology_graph` /
    `mag/src/orthoparagroups/`)
  run: |
    # After /split-generate wires type=paralogs_only:
    python -m src.pipeline.split_predict \
      --outdir output/paralogs_only_split \
      --type paralogs_only \
      --id-csv ready_legnet/ID.csv \
      --fold ready_legnet/fold.csv \
      --homology-edges mag/homology_graph/edges.tsv.gz \
      --homology-nodes mag/homology_graph/maps/nodes_extract.tsv \
      --seed 42
  notes: |
    Implemented by /split-generate (2026-08-01). Does not materialize SPLIT/.
    Unmapped IDs → test/val only (Locked). Distinct from mag orthoparagroups
    FASTA extract and from pangenome / mmseqs / blastp SBS clustering.
    Edge relation vocabulary: ortholog | paralog only.

# References

- Homology graph build: `mag/homology_graph/README.md`,
  `python -m src.run.homology_graph.build_mammals11`
- Species intersection / Compara availability:
  `mag/intersection.md`, `mag/homology_availability_report.md`
- Prior orthoparagroup C++ (related graph, different objective):
  `mag/src/orthoparagroups/`
- Split-generate / pipeline contracts: `wiki/split-generate.md`,
  `wiki/architecture.md`
- Ensembl Compara homology TSV dumps (release 116 `protein_default`)
