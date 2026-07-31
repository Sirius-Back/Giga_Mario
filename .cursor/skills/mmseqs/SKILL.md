---
name: mmseqs
description: >-
  Define and implement the MMseqs2 homology split (splits/mmseqs.md): MARKED
  DNA → MMseqs2 cluster-first folds → train/test/val at locked 60:20:20, then
  hand off to LegNet via /train. Use when the user mentions mmseqs, MMseqs2
  split, sequence-identity folds, easy-cluster splits, or /split-generate
  type=mmseqs.
disable-model-invocation: true
---

# MMseqs2 split (`splits/mmseqs.md`)

## Purpose

Caption + implementation contract for **MMseqs2 sequence-identity** fold
assignment. `/split-generate` reads [`splits/mmseqs.md`](../../../splits/mmseqs.md)
and writes `src/splits/mmseqs.py` for `/split` (`split-predict type=mmseqs`).
After `SPLIT/` materializes, continue with **LegNet** (`/train train=legnet`).

Follow: **validation-first**, **reproducibility**, **missing-data-policy**,
**scientific-integrity**, **artifact-registry**, **method-decision-tracking**,
**skills-write-and-exec-src**, **local-job-queue**.

Do **not** reimplement SBS assign/strat/materialize or LegNet training in chat.
Reuse `src.splits.sbs` + `src.pipeline.split_predict` / `split` + `/train`.

## Locked ratios (user)

standard split (train/val/test) = 60:20:20

Project weight order is always **train:test:val** → pass
`ratios=(0.6, 0.2, 0.2)` (equivalently `60:20:20`). Do not use Caduceus
~81/9/10 defaults for this strategy unless the user unlocks them.

## Obligatory caption

**Always** treat `splits/mmseqs.md` as the user-facing and generate-facing
spec. If missing or incomplete, stop and ask — do not invent roles or ratios.

## Inputs

| Input | Role |
|-------|------|
| **MARKED/** (or `--fna`) | Per-region DNA FASTA (panel MARKED; one record per ID) |
| **fold.csv** | Optional ZSV / fold filter |
| **stratification.csv** | Optional; SBS fold-grain strat |
| **ID.csv** | Optional; PCA genome colors |
| **mmseqs** binary | Required on PATH (or explicit `mmseqs_bin`) |
| **min_seq_id** / sensitivity | Clustering params — record in method-decision when chosen |

Primary sequence source is panel **`MARKED/`** (same as `gc` / `hashfrag`),
not LegNet-stitched `PARSED` adapters.

## Processing (must match caption)

```
(1) load MARKED / FNA (region ID = filename stem)
(2) hold out ZSV from fold.csv
(3) mmseqs easy-cluster (cluster-first; not dense all-vs-all for production)
(4) cluster → fold; SBS fold→train/test/val at ratios=(0.6, 0.2, 0.2)
(5) write split.csv (+ assignment + optional PCA diagnostics)
(6) /split materialize SPLIT/
(7) /train LegNet on that SPLIT/
```

### Step map → code

| Step | Action | Reuse |
|------|--------|-------|
| **(1) FNA** | Load MARKED dir | `src.splits.sbs.fna_io` |
| **(2) ZSV** | Hold out zsv set | SBS / `fold.csv` contracts |
| **(3) MMseqs2** | `easy-cluster` → cluster membership | `mmseqs` CLI; thin backend under `src.splits.sbs.backends` |
| **(4) assign** | Fold-grain train/test/val **60:20:20** | SBS `assign_from_features` (or cluster→fold glue) + `train_test_val_weights` |
| **(5) split.csv** | `ID\|train_test\|fold` | `assignment_rows_to_split_csv` |
| **(6) materialize** | `SPLIT/` | `/split` / `src.pipeline.split` |
| **(7) LegNet** | Fine-tune | `/train` (`train=legnet` / `human_legnet`) |

### Dense distance matrix — legacy only

`MMseqsDistanceBackend` (`easy-search` → `1 − pident/100`) is **small-n /
legacy**. Production clustering must **not** require a dense \(n\times n\)
matrix (`wiki/sbs.md`). Prefer **cluster-first** `easy-cluster` (cluster TSV →
folds). Keep the distance backend for smoke tests / tiny panels only.

## /split-generate checklist

```
mmseqs split-generate:
- [ ] splits/mmseqs.md present (this skill + caption)
- [ ] method-decision: ratios Locked 60:20:20; min_seq_id / -s choice
- [ ] WRITE cluster-first MMseqs backend (or extend mmseqs.py beyond DistanceMatrix)
- [ ] WRITE src/splits/mmseqs.py + wire type=mmseqs
- [ ] Default ratios=(0.6, 0.2, 0.2) in strategy / split_predict
- [ ] Reuse SBS assign/viz; do not fork materialize
- [ ] pytest for novel cluster → fold / ratio path
- [ ] artifact-registry + DONE for /split-generate
```

## /split → LegNet checklist

```
mmseqs → LegNet:
- [ ] split-predict type=mmseqs → split.csv (counts ~60/20/20 train/test/val)
- [ ] /split materialize SPLIT/ (PARSED + PREDICT trees)
- [ ] Pre-flight: train/val/test non-empty; mmseqs version logged
- [ ] /train train=legnet (or human_legnet) on that SPLIT/
- [ ] metrics.md + train-viz; register outs
```

after that split we will run LegNet

Do not stop at `split.csv` when the user asked for the full path — continue to
`/split` materialize then `/train` LegNet (unless they explicitly ask
split-only).

## Pre-flight

1. `mmseqs version` succeeds (or explicit binary path).
2. `MARKED/` exists with non-empty `*.fa` / `*.fasta` (or `--fna`).
3. Ratios fixed to **60:20:20** unless user unlocks otherwise.
4. Missing binary / MARKED / panel → **stop** (missing-data-policy).
5. Large panels → `queue.md` (`cpu_ram_heavy`; declare `peak_ram_gib`).

## Rules

1. Ratios Locked: train/val/test = 60:20:20 → `ratios=(0.6, 0.2, 0.2)` train:test:val.
2. Cluster-first MMseqs for production; no dense all-vs-all clustering on full panels.
3. Do not run MMseqs on LegNet-stitched PARSED sequences by default.
4. Clustering / QC / stratification: **SBS in-built** after cluster→fold.
5. Novel `./src` → pytest before COMPLETED.
6. After materialize → **LegNet** via `/train` (this skill’s default downstream).

## Coordination

| Path / skill | Role |
|--------------|------|
| `splits/mmseqs.md` | Spec caption (users + `/split-generate`) |
| `/split-generate` | Emits `src/splits/mmseqs.py` from caption |
| `/split` | split-predict + materialize `SPLIT/` |
| `/train` | LegNet fine-tune on materialized trees |
| `src.splits.sbs.backends.mmseqs` | Existing distance backend (legacy/small-n) |
| `wiki/sbs.md` | SBS C1/C2 contracts |
| Peer | `splits/hashfrag.md` (DNA homology), `splits/gc.md` / `splits/kmer.md` (composition SBS) |

## Additional resources

- Caption: [`splits/mmseqs.md`](../../../splits/mmseqs.md)
- Split generate: [`../split-generate/SKILL.md`](../split-generate/SKILL.md)
- Train / LegNet: [`../train/SKILL.md`](../train/SKILL.md)
- SBS: [`wiki/sbs.md`](../../../wiki/sbs.md)
- MMseqs2 docs: https://github.com/soedinglab/MMseqs2
