---
name: split
description: >-
  Universal split-predict + split: write src/run/run_id/{data}_{split}_{direct|adversarial}.py,
  exec it to assign folds and materialize SPLIT/. Use for /split. Not the legacy ready/M1/M2 path.
disable-model-invocation: true
---

# Split (`/split`)

## Purpose

Run the **universal** assign + materialize pipeline:

1. **split-predict** → `{outdir}/split.csv` (`ID|train_test|fold`)
2. **split** → `{outdir}/SPLIT/` (+ optional ZSV trees)

**Primary path** = `src.pipeline.split_predict` + `src.pipeline.split` (and fold/strat helpers / `id_rule` when needed).

Do **not** treat legacy `raw/`+`ready/` → `src.splits.main` M1/M2 as the primary flow. That path remains for Caduceus-legacy consumers only; strategy *generation* belongs to `/split-generate`.

## Obligatory inputs

Stop (missing-data-policy) if any required item is absent or empty.

| Input | Required | Role |
|-------|----------|------|
| **data id** | **yes** | Short slug for the panel (used in run script filename) |
| **split strategy id** | **yes** | Must match `splits/<id>.md` and a registered pipeline/`src.splits` type (e.g. `random`) |
| **ID.csv** | **yes** | Panel ID table (`genome\|…\|ID`); join key for assignment |
| **PREDICT** | **yes** | Root with `PREDICT/` (or the PREDICT dir itself) + `predict.csv` |
| **PARSED** | **yes** | Root with `PARSED/` (or the PARSED dir itself) |
| **outdir** | **yes** | Writable output root for `split.csv`, `SPLIT/`, optional ZSV |
| **mode** | **yes** | `direct` or `adversarial` (encoded in run script name) |
| **seed** | **yes** (default `42`) | Deterministic assignment |
| **intersect_allow** | **yes** (default `T`) | `T` skip missing PARSED/PREDICT; `F` raise |
| **fold.csv** | optional | Pre-assigned folds / ZSV holdouts (`ID\|fold`) |
| **stratification.csv** | optional | `ID` + strat columns (composite key in split-predict) |
| **sample_id** | optional | Filter mapped PREDICT rows when panel is sample-mapped |
| **run_id** | optional | Subdir under `src/run/run_id/` (default `run_id` package layout as-is) |

If strategy needs fold/strat CSVs that do not exist, build them via `generate_fold` / (when implemented) `generate_stratification` + `id_rule` — or stop and ask. Do **not** invent fold or strat labels.

## Code-first contract (LOCKED)

```
/split:
  1. Validate obligatory inputs (paths exist, non-empty; strategy caption present)
  2. WRITE exact runner → src/run/run_id/{data}_{split}_{direct|adversarial}.py
  3. EXEC that file (python path/to/that.py … or python -m after package wiring)
  4. DONE only after exec succeeds (split.csv + SPLIT/ on disk)
```

- Filename encodes **data**, **split strategy**, and **mode**:
  - `src/run/run_id/prok_random_direct.py`
  - `src/run/run_id/prok_random_adversarial.py`
- Runner must **call** pipeline APIs — never reimplement assignment or materialization in-chat or inline duplicates of `src/pipeline` logic.
- Must **not** break existing `src/pipeline` contracts.
- **Novel** edits under `src/` (new helpers, strategy hooks) → add/extend **pytest** under `tests/pipeline/` before declaring done.

## Mode

| Mode | Behavior |
|------|----------|
| `direct` | `run_split_predict` → `run_split` on the given ID/PREDICT/PARSED panel |
| `adversarial` | Optionally `run_adversarial` to copy panel into a new outdir for a subsequent target cycle, then `run_split_predict` → `run_split` on the adversarial panel (same contracts) |

## Reuse map (call these)

| Stage | Module | Entry |
|-------|--------|-------|
| Assign | `src.pipeline.split_predict` | `run_split_predict(...)` |
| Materialize | `src.pipeline.split` | `run_split(...)` |
| Fold table | `src.pipeline.generate_fold` | `run_generate_fold(...)` + `id_rule` |
| Strat table | `src.pipeline.generate_stratification` | stub — **NotImplemented**; do not invent |
| ID remap | `src.pipeline.id_rule` | `run_id_rule(...)` |
| Adversarial copy | `src.pipeline.adversarial` | `run_adversarial(...)` |
| Random helpers | `src.splits.random` | imported **inside** split-predict already |

Strategy algorithms live under `src/splits/<id>.py` (produced by `/split-generate`). `/split` only wires them through pipeline `type=` / registered helpers.

## Runner skeleton

Write something equivalent (paths/params filled from obligatory inputs):

```python
"""Auto-generated /split runner — do not reimplement pipeline logic here."""
from pathlib import Path

from src.pipeline.split_predict import run_split_predict
from src.pipeline.split import run_split
# from src.pipeline.adversarial import run_adversarial  # when mode=adversarial

DATA = "prok"
SPLIT = "random"
MODE = "direct"  # or adversarial

def main() -> None:
    outdir = Path("output/prok_random_direct")
    split_csv = run_split_predict(
        outdir=outdir,
        type=SPLIT,
        seed=42,
        id_csv=Path("path/to/ID.csv"),
        fold_csv=None,              # optional
        stratification_csv=None,    # optional
    )
    run_split(
        split_csv,
        parsed_target=Path("path/to/PREDICT_parent_or_PREDICT"),
        parsed_data=Path("path/to/PARSED_parent_or_PARSED"),
        outdir=outdir,
        strategy="traintestval",
        intersect_allow=True,
        sample_id=None,             # optional mapped filter
        id_csv=Path("path/to/ID.csv"),
    )

if __name__ == "__main__":
    main()
```

## Exec

```bash
python src/run/run_id/{data}_{split}_{direct|adversarial}.py
```

Prefer project conda env used for pipeline pytest. Heavy panels → SLURM wrapper calling the **same** file.

## Outputs

| Artifact | Content |
|----------|---------|
| `{outdir}/split.csv` | `ID\|train_test\|fold` |
| `{outdir}/SPLIT/FASTA/{TRAIN\|TEST\|VAL\|…}/` | PARSED sequences |
| `{outdir}/SPLIT/PREDICT/{…}/` | PREDICT + per-bucket `predict.csv` |
| `{outdir}/PREDICT\|PARSED/zero-shot-validation/` | ZSV holdouts (not used in train) |

## Workflow checklist

```
split:
- [ ] Collect obligatory inputs; stop if any missing
- [ ] Confirm splits/<id>.md exists (strategy caption)
- [ ] If fold/strat needed and missing: generate_fold / ask (strat stub)
- [ ] WRITE src/run/run_id/{data}_{split}_{mode}.py
- [ ] EXEC that file
- [ ] Verify split.csv + SPLIT/ (+ ZSV if fold has zsv)
- [ ] Novel src changes → pytest; method-decision + artifact-registry
```

## Rules

- Write runner **before** exec; exec **before** COMPLETED
- Never invent TPM, fold, or strat labels
- Never change `src/pipeline` I/O contracts silently
- Fixed seeds; relative project paths
- Legacy `python -m src.splits.main --raw --ready` is **not** the `/split` primary path

## Coordination

| Path / skill | Role |
|--------------|------|
| `/split-generate` | Author `src/splits/<id>.py` + register |
| `wiki/architecture.md` | Stage contracts |
| `wiki/split-generate.md` | Strategy generation docs |
| `@train` / `src.pipeline.train` | Consumes `SPLIT/` |

## Additional resources

- Architecture: [`wiki/architecture.md`](../../../wiki/architecture.md)
- Strategy captions: [`splits/`](../../../splits/)
- Generate strategies: [`../split-generate/SKILL.md`](../split-generate/SKILL.md)
