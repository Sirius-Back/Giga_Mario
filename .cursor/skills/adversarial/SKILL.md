---
name: adversarial
description: >-
  Build adversarial panel via src.pipeline.adversarial, then random split
  (split-predict + split); write/exec
  src/run/<run_id>/{data}_{split}_adversarial.py. Use for /adversarial.
disable-model-invocation: true
---

# Adversarial (`/adversarial`)

Combine / copy a real panel into a new adversarial outdir, then run a **random** split so downstream `/train` can use `mode=adversarial`. **Must reuse `./src`** — no parallel reimplementation.

## Obligatory inputs

| Input | Meaning |
|-------|---------|
| **`run_id`** | Run directory under `src/run/<run_id>/` |
| **`data`** | Data panel id (filename stem) |
| **`split`** | Split strategy id — **`random` only** for this skill |
| **`outdir_new`** | Destination adversarial panel root (must differ from source) |

**Source panel** — one of:

| Mode | Required |
|------|----------|
| **A — prior outdir** | `outdir` containing `split.csv`, `PREDICT/`, `PARSED/` |
| **B — explicit** | `split_csv` + `parsed_target` (PREDICT) + `parsed_data` (PARSED) |

After combine, **random split** also needs (for materialize):

| Input | Meaning |
|-------|---------|
| **`id_csv`** (or IDs from panel) | For `src.pipeline.split_predict` when rebuilding assignments |
| **`PARSED` / `PREDICT`** | Already copied into `outdir_new` by adversarial |
| **`seed`** | Default `42` |
| **`split` materialize strategy** | e.g. `traintestval` |
| **`intersect_allow`** | `T` (default) or `F` |

If obligatory inputs missing → **stop**. Do not invent PREDICT/PARSED or fold labels.

## Code-first contract (LOCKED)

```
/adversarial cycle:
  1. Validate source panel (outdir OR split_csv+PREDICT+PARSED)
  2. WRITE src/run/<run_id>/{data}_{split}_adversarial.py
     — imports src.pipeline.adversarial + src.pipeline.split_predict + src.pipeline.split
  3. EXEC that script:
       a. run_adversarial → outdir_new
       b. random split-predict → split.csv
       c. split materialize → SPLIT/ (+ ZSV if fold.csv marks zsv)
  4. REUSE the same script later (no agent required for re-run)
```

**Never** reimplement panel copy or fold assignment in-chat.

## Script path

```
src/run/<run_id>/{data}_{split}_adversarial.py
```

Example: `src/run/exp01/prok_random_adversarial.py`

## What the run script must do

1. `src.pipeline.adversarial.run_adversarial(...)` → hardlink/copy real `PREDICT`/`PARSED`/`split.csv` into `outdir_new`.
2. **Random** `/split` via pipeline stages only:
   - `src.pipeline.split_predict.run_split_predict(..., type="random", …)`
   - `src.pipeline.split.run_split(...)` (materialize `SPLIT/`; ZSV aside when present)
3. Do **not** call `@adapt` / parse stages unless the user explicitly locked a full re-parse cycle.
4. Emit paths consumed by `/train` with `mode=adversarial`.

## Exact patterns

```bash
conda run -n caduceus_env python src/run/<run_id>/<data>_<split>_adversarial.py

# Underlying stages (debug only)
python -m src.pipeline.adversarial --outdir-new <new> --outdir <src_panel>
python -m src.pipeline.split_predict --type random --outdir <new> --seed 42 …
python -m src.pipeline.split --split-csv <new>/split.csv \
  --parsed <new>/PARSED --predict <new>/PREDICT --outdir <new> …
```

## Workflow checklist

```
adversarial:
- [ ] Confirm run_id, data, split=random, outdir_new ≠ source
- [ ] Confirm source panel complete (split.csv + PREDICT + PARSED)
- [ ] Write src/run/<run_id>/{data}_{split}_adversarial.py
- [ ] Exec: adversarial combine → random split-predict → split
- [ ] Verify outdir_new/{PREDICT,PARSED,split.csv,SPLIT}/
- [ ] method-decision + artifact-registry
```

## Coordination

| Module | Role |
|--------|------|
| `src.pipeline.adversarial` | Panel combine / hardlink copy |
| `src.pipeline.split_predict` | Random `split.csv` |
| `src.pipeline.split` | Materialize SPLIT + ZSV trees |
| `/train` | Train with `_adversarial.py` filename mode |
| `/pipeline` | May invoke this skill when adversarial is specified |

## Rules

- Reuse `./src` only — especially `src.pipeline.adversarial` + split stages
- Filename ends with `_adversarial.py` (no train model segment)
- Fixed seed; relative paths
- Register outputs in `docs/artifact-registry.md`
