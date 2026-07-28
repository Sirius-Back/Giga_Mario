---
name: prepare-target
description: >-
  Execute parse_target: raw TARGET folder + ID.csv → PREDICT/ID.ext + predict.csv
  via src/pipeline/parse_target.py (Caduceus/LegNet label contracts). Write-and-exec.
disable-model-invocation: true
---

# Prepare Target (`parse_target`)

## Goal

```
{genome}.csv TARGET folder + ID.csv  ──►  src/pipeline/parse_target.py  ──►  PREDICT/
                                                                          ├── {id}.ext
                                                                          └── predict.csv
```

Todo-orchestrator skill remains `@prepare` (planning). This skill runs **parse_target** only.

## Command

```bash
python -m src.pipeline.parse_target \
  --target path/to/tpm_or_genome_csvs \
  --id-csv path/to/ID.csv \
  --outdir path/to/out \
  --to-type caduceus
```

`--to-type` ∈ {`caduceus`,`legnet`}. Missing gene → predict **0**.

## Tests

```bash
python -m pytest tests/pipeline/test_pipeline_io.py -q -k parse_target
```
