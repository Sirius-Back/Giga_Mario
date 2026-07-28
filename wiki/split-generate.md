# Split Generate

**Skill:** `/split-generate` → [`.cursor/skills/split-generate/SKILL.md`](../.cursor/skills/split-generate/SKILL.md)  
**Related:** [`/split`](../.cursor/skills/split/SKILL.md) (assign + materialize), [`architecture`](architecture.md) (pipeline contracts)

## Purpose

Generate **split-predict strategy implementations** from captions in `splits/*.md`.

Output is code under `src/splits/<id>.py` (plus registry / pipeline `type=` hooks), not a finished `SPLIT/` tree. Materialization is `/split` (`split-predict` → `split`).

## Obligatory inputs

| Input | Required | Notes |
|-------|----------|-------|
| Split strategy id | yes | e.g. `random` |
| `splits/<id>.md` | yes | Description, Split roles, Implementations, References |
| Algorithm intent | yes | From MD + any user locks (holdouts, strata, ratios) |
| Reuse preference | yes (default) | Prefer `src/splits/` + `src/pipeline/` helpers |
| Fold rules | when needed | `prepare_fold.csv` → `generate_fold` via `id_rule` |
| Stratification rules | when needed | Provide `stratification.csv`, or wait — auto-generate is stubbed |
| Tests | for novel logic | pytest under `tests/pipeline/` (or splits tests) |

## Outputs

| Artifact | Role |
|----------|------|
| `src/splits/<id>.py` | Assignment helpers / strategy runner pieces |
| `src/splits/main.py` registry and/or `split_predict` `type=` | Discovery |
| `fold.csv` helpers | Via `src.pipeline.generate_fold` + `id_rule` (not duplicated) |
| `stratification.csv` | User-supplied, or future `generate_stratification` |
| pytest | Required when behavior is new |
| `method-decision.md` | Algorithm choice + confidence |

## How `splits/*.md` maps to code

```text
splits/<id>.md
  frontmatter id/aliases     →  module + registry keys
  # Description              →  docstring / justification
  # Split                    →  train/val/test/zsv labels
  # Implementations          →  ratios, seeds, reference behavior
  # References               →  citations in decisions (no invented DOIs)
        │
        ▼
src/splits/<id>.py  ──imports──►  src.splits.common / random helpers
        │
        ▼
src.pipeline.split_predict (type=<id>)  →  split.csv
        │
        ▼
/split runner  →  src.pipeline.split  →  SPLIT/
```

Complex architectures (CV folds, chromosome/species holdout, adversarial panels) are allowed when the caption and locks require them. Do not invent unspecified rules.

## Fold (`split_fold`)

1. Rules file: `identificator|column|fold` (see `generate_fold`).
2. Resolve with `id_rule`: `id_col_1=column` → `id_col_2=ID`.
3. Write `fold.csv` via `run_generate_fold`.
4. Labels `zsv` / aliases → zero-shot holdout; excluded from random train/val/test; materialized under `zero-shot-validation/`.

## Stratification (`split_stratification`)

1. Table: `ID` + strat columns; `split_predict` uses a **composite** key across non-ID columns.
2. Prefer an existing `stratification.csv`.
3. `src.pipeline.generate_stratification` is a **NotImplemented** stub (warns and raises). Do not fabricate strat labels. When implemented, follow the same `id_rule` pattern as `generate_fold`.

## Reuse rules

- Import `assign_folds_random` / `assign_folds_stratified` from `src.splits.random` when ratios match Caduceus-aligned random.
- Call `run_id_rule` / `run_generate_fold` instead of copying remap loops.
- Do not break `src/pipeline` I/O contracts; extend with defaults that preserve legacy behavior.
- Novel code → pytest before claiming the strategy ready for `/split`.

## Separation from `/split`

| Skill | Writes | Executes |
|-------|--------|----------|
| `/split-generate` | `src/splits/<id>.py` (+ registry) | pytest / smoke import |
| `/split` | `src/run/run_id/{data}_{split}_{direct\|adversarial}.py` | that runner → `split.csv` + `SPLIT/` |

## Commands (after generation)

Strategy unit tests (example):

```bash
python -m pytest tests/pipeline/test_split_predict.py -q
```

Then assign + materialize with `/split` (runner under `src/run/run_id/`).
