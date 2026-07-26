---
name: split
description: >-
  Write split strategy code under src/splits/, execute it on raw/+ready/, produce
  train/val/test folds (M1 TPM + optional M2). Use for /split or region folds.
disable-model-invocation: true
---

# Split

## Purpose

Produce reproducible **train / validation / test** folds for a **named split strategy** (`splits/*.md`) by:

1. **Writing** strategy code under `src/splits/<id>.py`
2. **Executing** it via `python -m src.splits.main --strategy <id> …`
3. **Only then** treating fold trees as ready for `@caduceus` / downstream

**Atomic unit** = ready sample (DNA window + linked prediction). Default prediction = continuous **TPM**.

Do **not** re-convert `ready/` during split. If a strategy needs genomic metadata, read `raw/` to build assignment tables, then **apply** those assignments to ready files (hardlink/symlink sequences).

## Required inputs

| Input | Path | Role |
|-------|------|------|
| **raw** | `raw/` | FNA/GTF/TPM/mapping — for strategies that need genomic metadata |
| **ready** | `ready/` (or `data_ready/`) | Already prepared windows + TPM — **do not prepare/adapt here** |
| **сплит** | `splits/<id>.md` | Strategy caption |

If raw or ready is missing → **stop**. Do not invent data. Do not run `@adapt` inside split when `ready/` already exists.

Optional: `--seed` (default 42), `--out` (default `splits/<id>`), `--max-samples` (smoke).

## Code-first contract (LOCKED)

```
Split skill cycle:
  1. Read splits/<id>.md caption
  2. WRITE or UPDATE src/splits/<id>.py (+ register in src/splits/main.py)
  3. EXEC: python -m src.splits.main --strategy <id> --raw raw --ready ready --seed 42
  4. RUN complete only after exec succeeds (manifests + folds on disk)
```

**Never** invent fold membership in-chat. **Never** reimplement windowing in the split skill — that is `@adapt` / `src/preprocessing.py`.

New strategies: add `src/splits/<id>.py` with `run_<id>_split(...)`, register in `STRATEGY_RUNNERS` inside `src/splits/main.py`.

## Dual-stream (ready panel)

| Stream | Source | Notes |
|--------|--------|-------|
| Genomic | `ready/ready.csv` + `ready/caduceus_ready/**/sequences/*.txt` | Split IDs only; no reconversion |
| Prediction M1 | TPM column | Continuous |
| Prediction M2 (random) | M1 fold class `{train:0,val:1,test:2}` | Stratified by M1 |

Linkage: every sample_id has exactly one prediction. Non-coding / missing gene TPM already **0** from ready prep.

## Random strategy outputs (`splits/random/`)

| Path | Content |
|------|---------|
| `M1/{train,val,test}/` | Sequences + `labels.tsv` (**TPM**); `ready.csv` subset |
| `M2/{train,val,test}/` | Same sequences; labels = **M1 fold** (stratified assignment) |
| `splits_log.csv` | `data_input\|M1\|M2` |
| `M1/fold_manifest.tsv`, `M2/fold_manifest.tsv` | Membership + paths |
| `metadata.json` | Seed, counts, encoding |

Ratios (Caduceus-aligned): ~10% test; of remainder ~10% val / ~90% train. Seeds: M1=`seed`, M2=`seed+1` (stratified by M1).

## Exact command

```bash
conda run -n caduceus_env python -m src.splits.main \
  --strategy random \
  --raw raw \
  --ready ready \
  --seed 42
```

Smoke:

```bash
python -m src.splits.main --strategy random --max-samples 500 --out splits/random_smoke
```

Heavy panels → wrap in sbatch (even CPUs, mem, time, logs).

## Analyze split caption (mandatory before writing code)

Read `splits/<id>.md`:

| Section | Extract |
|---------|---------|
| `# Description` | What is assigned |
| `# Split` | train / validation / test / zero_shot roles |
| `# Implementations` | urls / notes / fallbacks |

Map caption → `src/splits/<id>.py` behavior. Record Locked choices in `method-decision.md`.

## Workflow checklist

```
split:
- [ ] Parse strategy id + confirm raw/ + ready/ present
- [ ] Read splits/<id>.md caption
- [ ] Write/update src/splits/<id>.py + main.py registry
- [ ] Exec python -m src.splits.main --strategy <id> …
- [ ] Verify M* folds + splits_log / manifests
- [ ] docs/split-report.md + method-decision + artifact-registry
```

## Rules

- Write code in `src/splits/` **before** exec; exec **before** declaring done
- Never invent TPM or fold labels
- Never convert/adapt ready files during split
- Use `raw/` only to build assignment tables when the strategy requires it
- Fixed seeds; relative project paths
- Zero-shot only when explicitly indicated in the split MD / user lock

## Coordination

| Skill / path | Role |
|--------------|------|
| `src/splits/main.py` | Dispatcher |
| `src/splits/random.py` | Random M1/M2 |
| `src/preprocessing.py` / `@adapt` | Build `ready/` **before** split |
| `splits/*.md` | Strategy captions |
| `@caduceus` | Consumes fold `sequences/` + `labels.tsv` |

## Additional resources

- Caption: [`splits/random.md`](../../splits/random.md)
- Ready prep: [`wiki/conversion.md`](../../wiki/conversion.md)
