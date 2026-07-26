# Split: raw + ready → folds

**Code:** `src/splits/` · **Entry:** `python -m src.splits.main --strategy <id>`

## Inputs

- `raw/` — genomes/annotations/TPM (for strategies that need metadata)
- `ready/` → `data_ready/` — prepared windows (**not** re-converted)

## Random outputs (`splits/random/`)

| Tree | Prediction |
|------|------------|
| `M1/{train,val,test}/` | TPM |
| `M2/{train,val,test}/` | M1 fold class (stratified) |
| `splits_log.csv` | `data_input\|M1\|M2` |

Skill workflow: **write** `src/splits/<id>.py` → **exec** main → **run** complete.
