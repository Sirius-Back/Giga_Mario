---
id: gcn
name: GCN / VGAE graph split
aliases:
  - GCN
  - vgcn
  - gcn_split
  - vgae_label
---

# Description

Assign train / validation / test from a **graph neural** labeling produced by a
named **VGCN / VGAE** (or compatible) model. The caller passes a **model name**
or a short **model description**; the strategy resolves that to a concrete run
under `VGAE/` (or a future model registry) and emits `split.csv`
(`ID|train_test|fold`).

Resolution order (do **not** skip ahead if an earlier step succeeds):

1. **Reuse labeling** — if the named model already has a ready role table
   (`split.csv`, or equivalent soft/hard scores + ID map), use it.
2. **Infer from trained weights + data** — if a checkpoint / pack exists for
   that model (e.g. `VGAE/<model>/checkpoints/best.pt` + `pack/`) but no
   labeling yet, run inference on the panel graph and size-constrain to
   train:test:val.
3. **Train then label** — if neither labeling nor usable weights exist for the
   requested model, train the VGCN/VGAE on the panel graph (same contracts as
   `splits/vgae.md`: compositional GC/k-mer + edge weights into the GCN;
   homology **not** in the encoder), then assign roles.

Default role counts: **3:1:1** (train:test:val). Homology objective and
`sd_random` checker remain as in VGAE (`L_hom` post-assign only).

Models may be **already trained** (e.g. `stage1_region_k5`, `stage2_hash_k5`)
or **ones you train later** under the same naming scheme; the caption does not
hard-code a closed model list.

# Split

train:
- Regions (or pooled sequence IDs) whose model role is `train`.

validation:
- Regions whose model role is `val`.

test:
- Regions whose model role is `test`.

zero_shot:
- Optional ZSV from `fold.csv` (held out before assign when provided).

# Inputs

| Input | Required | Role |
|-------|----------|------|
| **model** | **yes** | Short name **or** free-text description resolving to a VGAE/VGCN run (e.g. `stage1_region_k5`, `stage2_hash_k5`, `VGAE/stage1_region_k5`, or “region k=5 classic VGAE”) |
| **ID.csv** / panel | yes (for materialize) | Join key for `/split` |
| **PARSED** / **PREDICT** | yes (for materialize) | `/split` materialize roots |
| **outdir** | yes | Writes `split.csv` (+ reuses or creates under `VGAE/<model>/` when training/inferring) |
| **seed** | yes (default `42`) | Deterministic size-constrained assign |
| **ratios** | optional | Default `(3,1,1)` train:test:val |

Model resolution hints (implementation):

- Exact directory: `VGAE/<model>/` if it exists.
- Alias / description: match `train_meta.json` fields (`grain`, `k`, `stage`)
  or caption notes under `VGAE/*/train_meta.json`.
- If ambiguous → **stop** (missing-data-policy); do not invent a model.

# Pipeline

```
model name | description
        │
        ▼
  resolve → VGAE/<model>/  (or registered path)
        │
        ├─ has split.csv / role scores? ──yes──► use labeling
        │
        ├─ has checkpoint + pack + data? ─yes──► infer → labeling
        │
        └─ else ──► train VGCN/VGAE → infer → labeling
        │
        ▼
   size-constrain 3:1:1 (unless ratios overridden)
        │
        ▼
   split.csv  →  /split materialize → SPLIT/
```

# Implementations

- name: GigaMario GCN/VGAE labeling split
  url: local toolkit
  paper: Kipf & Welling, VGAE (2016)
  split_location: |
    Caption: `splits/GCN.md`.
    Train/infer core: `src/splits/vgae/` (classic VGAE; future VGCN variants
    plug in as additional model backends under the same `VGAE/<model>/` layout).
    Pipeline hook: `src/pipeline/split_predict.py` (`type=gcn` / `type=vgae`
    as wired).
  run: |
    # Prefer an already-labeled run
    #   model=stage1_region_k5  →  reuse VGAE/stage1_region_k5/split.csv

    # Infer-only when checkpoint+pack exist but split.csv missing:
    #   python -m src.splits.vgae --stage 1 --out VGAE/stage1_region_k5 ...

    # Full cascade via /split (when type=gcn is wired):
    #   run_split_predict(..., type="gcn", model="stage2_hash_k5", ...)
    #   → split.csv → run_split(...)

    # Existing trained models (examples; not exhaustive):
    #   stage1_region_k5   — region–region contingency + GC/k-mer wrap
    #   stage2_hash_k5     — hash-node graph, pooled to regions
  notes: |
    Homology firewall unchanged: OG/PG never enter GCN/VAE encoder.
    Early stop for any new train: min_epochs=25, patience=10 on hard L_hom.
    Outputs stay under `VGAE/<model>/` (pack, checkpoints, logs, tensorboard,
    figures, split.csv) so later `/split` / train-viz / checkers can consume them.
    New models you train later should use the same directory contract so this
    caption keeps working without editing.

# References

- `splits/vgae.md` — classic VGAE Stage1/Stage2 contracts
- `splits/pangenome.md` — underlying pangenome graph construction
- Kipf & Welling, Variational Graph Auto-Encoders
- `wiki/architecture.md`, `/split` skill
- `split-check-othoparagroup` (`sd_random`)
