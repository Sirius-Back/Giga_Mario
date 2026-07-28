---
name: split-generate
description: >-
  Generate split-predict strategy implementations from splits/*.md into
  src/splits/<id>.py (fold/strat helpers, id_rule). Use for /split-generate.
  Does not run the full split-predict+split materialize path (/split).
disable-model-invocation: true
---

# Split Generate (`/split-generate`)

## Purpose

**Generate** (write/update) split-predict **strategy** code so `/split` can assign folds via `src.pipeline.split_predict`.

This skill owns what used to be embedded “write `src/splits/<id>.py`” work inside the old `/split` skill. It does **not** materialize `SPLIT/` trees — that is `/split`.

## Obligatory inputs

Stop if any required item is missing.

| Input | Required | Role |
|-------|----------|------|
| **split strategy id** | **yes** | Target id; must match or will create `splits/<id>.md` only when user supplies/locks the caption |
| **splits/\<id\>.md** | **yes** | Spec caption: Description, Split roles, Implementations, References |
| **algorithm intent** | **yes** | How train/val/test/(zero_shot) are assigned (from MD + user locks) |
| **reuse targets** | **yes** (default prefer existing) | Prefer helpers in `src/splits/` + `src/pipeline/` over new code |
| **fold rules** | when ZSV / explicit folds needed | `prepare_fold.csv` pattern → `generate_fold` / `id_rule` |
| **stratification rules** | when stratified assignment needed | User-supplied `stratification.csv` **or** prepare_strat rules — see stub note |
| **tests expectation** | **yes** for novel logic | pytest covering new strategy / helpers |

Optional: seed defaults, ratio notes from Implementations, paper URLs (document only; do not invent data).

## What to produce

1. `src/splits/<id>.py` — assignment helpers + any strategy-specific APIs
2. Register in `src/splits/main.py` **and/or** wire into `src.pipeline.split_predict` (`type=<id>`) — prefer **reuse** of existing random helpers; extend `type=` only when needed
3. Pipeline-facing fold/strat helpers when the strategy requires related CSVs
4. pytest for novel behavior
5. method-decision entry when choosing among algorithms

## Spec → code map (`splits/*.md`)

| MD section | Maps to |
|------------|---------|
| Frontmatter `id` / `aliases` | Module name `src/splits/<id>.py`; registry keys |
| `# Description` | Docstring + method-decision Justification |
| `# Split` (`train` / `validation` / `test` / `zero_shot`) | Assignment labels → `train_test` (+ ZSV → `zsv`) |
| `# Implementations` | Ratios, seeds, reference behaviors; prefer Caduceus-aligned defaults when unspecified |
| `# References` | Cite in comments / method-decision; never invent DOIs |

Architecture may be complex (multi-fold CV, chromosome holdout, species holdout). Implement only what the caption + user lock require. Unresolved design → **Open** in method-decision; do not guess.

## Algorithms & reuse

**Prefer reuse:**

| Existing | Use for |
|----------|---------|
| `src.splits.random.assign_folds_random` | Unstratified train/val/test sizes (Caduceus-aligned ratios) |
| `src.splits.random.assign_folds_stratified` | Stratified assignment |
| `src.splits.common` | Shared fold sizing / I/O helpers when still applicable |
| `src.pipeline.split_predict.run_split_predict` | Consumer of strategy `type=` |
| `src.pipeline.id_rule.run_id_rule` | Remap prepare rules → ID lists |
| `src.pipeline.generate_fold.run_generate_fold` | `ID.csv` + `prepare_fold.csv` → `fold.csv` |

Do **not** duplicate assignment logic inside `/split` runners. Strategy code belongs under `src/splits/` (or thin pipeline wrappers that import it).

### `split_fold` logic

For strategies / panels that need **fold.csv** (including ZSV holdouts):

1. Accept `prepare_fold.csv` rows: `identificator|column|fold`
2. Resolve IDs via `run_id_rule([identificator], id_csv, id_col_1=column, id_col_2="ID")`
3. Call `run_generate_fold` (do not reimplement)
4. ZSV labels normalize to `zsv` (`generate_fold.is_zsv_fold` / `normalize_fold_label`)
5. `split_predict` excludes ZSV from random assignment; materialize moves them to `zero-shot-validation/`

### `split_stratification` logic

For related **stratification.csv**:

1. Expected shape: `ID` + one or more strat columns (composite key in split-predict)
2. Prefer user-provided `stratification.csv` when present
3. Auto-build via `generate_stratification` is currently a **NotImplemented stub** (`src.pipeline.generate_stratification`): warns `"Not implemented"` and raises — **do not invent** strat labels
4. When implementing the stub later: mirror `generate_fold` + `id_rule` (`id_col_1` → `ID`); add pytest

## Code-first contract (LOCKED)

```
/split-generate:
  1. Read splits/<id>.md (obligatory)
  2. Record algorithm choice in method-decision.md (Tentative/Locked)
  3. WRITE/UPDATE src/splits/<id>.py (+ registry / split_predict type hook)
  4. WRITE pytest for novel logic
  5. EXEC pytest (and optional smoke import)
  6. DONE — strategy available for /split (does not itself write SPLIT/)
```

## Workflow checklist

```
split-generate:
- [ ] Obligatory inputs present (strategy id + splits/<id>.md + algorithm intent)
- [ ] Analyze caption → assignment / fold / strat requirements
- [ ] Reuse src/splits + src/pipeline helpers; avoid fork copies
- [ ] Implement src/splits/<id>.py; register
- [ ] Wire fold via generate_fold/id_rule; strat only if CSV exists or stub lifted
- [ ] pytest green
- [ ] method-decision + artifact-registry
```

## Rules

- Generate strategy code; do **not** run full `/split` materialize unless user also asks `/split`
- Never invent fold/strat/TPM values
- Preserve proven `src.splits.random` / pipeline contracts unless user locks a cutover
- Novel src → pytest
- Fixed seeds in examples; relative paths

## Coordination

| Path / skill | Role |
|--------------|------|
| `/split` | Writes `src/run/run_id/{data}_{split}_{mode}.py` and execs split-predict+split |
| `splits/*.md` | Spec captions |
| `wiki/architecture.md` | Pipeline contracts |
| `wiki/split-generate.md` | Human mirror of this skill |

## Additional resources

- Human docs: [`wiki/split-generate.md`](../../../wiki/split-generate.md)
- Architecture: [`wiki/architecture.md`](../../../wiki/architecture.md)
- Example caption: [`splits/random.md`](../../../splits/random.md)
- `/split` skill: [`../split/SKILL.md`](../split/SKILL.md)
