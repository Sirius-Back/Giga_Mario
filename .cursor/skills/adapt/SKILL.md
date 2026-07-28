---
name: adapt
description: >-
  Universal adapt stage: GTF+FNA (+ ID.csv) → MARKED/*.fa + intersect.csv via
  src/pipeline/adapt.py. Write-and-exec only. Legacy Caduceus data_ready prep is @adapt-legacy.
disable-model-invocation: true
---

# Adapt

## Goal

Mark genomic intervals from GTF + FNA into model-agnostic **MARKED** FASTA and **intersect.csv**.

```
GTF/ + FNA/ + ID.csv  ──►  src/pipeline/adapt.py  ──►  outdir/
                                                      ├── MARKED/{id}.fa
                                                      └── intersect.csv
```

Does **not** join TPM (that is `parse_target` / `@prepare-target`). Does **not** assign folds (`split-predict`).

Legacy Caduceus `data_ready/` builder: **`@adapt-legacy`** → `src/preprocessing.py`.

--------------------------------------------------
GENERAL PRINCIPLES
--------------------------------------------------

Reproducible, deterministic, validation-first. Skills **write-and-exec** `./src` — do not reimplement marking in-chat. Do not change proven `src/preprocessing.py` behavior from this skill.

--------------------------------------------------
INPUTS
--------------------------------------------------

| Input | Role |
|-------|------|
| `--gtf` | Folder of `.gtf` |
| `--fna` | Folder of `.fna` / `.fa` |
| `--id-csv` | `ID.csv` from `id_gen` |
| `--task` | `gene` (default) or `promotor` |
| `--size` | Flank bp (`gene`) or window width (`promotor`) |
| `--outdir` | Output root |

--------------------------------------------------
OUTPUT
--------------------------------------------------

- `MARKED/{id}.fa` — header `>|genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID`
- `intersect.csv` — `ID1|ID2|intersection_size` (legacy analogue: `neighbours.csv`)

--------------------------------------------------
EXACT COMMAND
--------------------------------------------------

```bash
python -m src.pipeline.adapt \
  --gtf path/to/gtf \
  --fna path/to/fna \
  --id-csv path/to/ID.csv \
  --outdir path/to/out \
  --task gene \
  --size 10000
```

--------------------------------------------------
WORKFLOW
--------------------------------------------------

```
adapt:
- [ ] Confirm ID.csv exists (run id_gen if needed)
- [ ] Exec src/pipeline/adapt.py
- [ ] Verify MARKED/*.fa headers + intersect.csv columns
- [ ] Register artifacts; do not invent sequences
```

## Coordination

| Skill / module | Role |
|----------------|------|
| `src/pipeline/id_gen.py` | Build ID.csv |
| `src/pipeline/parse_fasta.py` | MARKED → PARSED |
| `src/pipeline/parse_target.py` | TARGET → PREDICT |
| `@adapt-legacy` | Old Caduceus `data_ready/` path |
| `wiki/architecture.md` | Contracts |

## Tests

```bash
python -m pytest tests/pipeline/test_pipeline_io.py -q
```
