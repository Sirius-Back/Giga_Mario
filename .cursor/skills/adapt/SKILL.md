---
name: adapt
description: >-
  Mandatory stage between raw genomic data and Caduceus: CDS±10kb DNA windows
  with neighbour trim, large-gene crop, matched non-coding, continuous TPM;
  write data_ready/ Caduceus-ready dataset via src/preprocessing.py. Never splits folds.
disable-model-invocation: true
---

# Adapt

## Goal

`adapt` prepares Caduceus-ready DNA windows + continuous TPM from raw FNA/GTF/TPM.

```
raw/{fna,gtf,tpm}  ──►  src/preprocessing.py  ──►  data_ready/
                                                      ├── ready.fna / ready.csv
                                                      ├── non_coding.csv, neighbours.csv, large_genes.csv
                                                      └── caduceus_ready/all/{sequences/*.txt, labels.tsv}
```

`@split` owns folds. This skill MUST NOT perform train/validation/test splitting.

--------------------------------------------------
GENERAL PRINCIPLES
--------------------------------------------------

Reproducible, deterministic (seed 42), restartable, idempotent, scientific, fully documented.

Never silently guess. Follow: validation-first, missing-data-policy, reproducibility, scientific-integrity, method-decision-tracking, artifact-registry, slurm-execution-policy, task-status.

--------------------------------------------------
STAGE 1 — REPOSITORY AUDIT
--------------------------------------------------

Expect:

```
raw/fna/*.fna(.gz)
raw/gtf/*.gtf(.gz)
raw/tpm/*.csv
raw/random_borzoi_expr_file_mappings.csv
```

Pair by GCF accession. Skip genomes missing a local TPM (do not invent). Abort if no complete bundles.

--------------------------------------------------
STAGE 2 — CADUCEUS FORMAT
--------------------------------------------------

See [docs/caduceus_format.md](../../docs/caduceus_format.md). Continuous TPM → `caduceus_ready/{fold}/sequences/*.txt` + `labels.tsv`.

--------------------------------------------------
STAGE 3 — WINDOW STRATEGY (LOCKED 2026-07-27)
--------------------------------------------------

| Rule | Value |
|------|-------|
| Anchor | CDS span (min–max CDS from GTF) |
| Flank | ±**10 000** bp |
| Neighbours | If other CDS enter the window → trim at neighbour CDS corner; log `neighbours.csv` |
| Large genes | CDS length > **130 000** → 10 kb before strand-aware start + **120 kb** of CDS; log `large_genes.csv` |
| Orientation | Forward genomic sequence only |
| Non-coding | Intergenic complement; match gene **length** & **GC** (greedy 1:1); TPM **0** |
| Properties | Gene + non-coding Length/GC → `non_coding.csv` |

No multi-window genes. No RC by default.

--------------------------------------------------
OUTPUT
--------------------------------------------------

Directory: **`data_ready/`** (see [wiki/conversion.md](../../wiki/conversion.md))

- `ready.fna` — `>Genome|GeneOrID|Chr|Position_start|Position_end`
- `ready.csv` — `Genome|GeneOrID|Chr|Position_start|Position_end|TPM`
- `non_coding.csv`, `neighbours.csv`, `large_genes.csv`
- `caduceus_ready/`, `statistics.json`, `metadata.json`

--------------------------------------------------
EXACT COMMAND
--------------------------------------------------

**Do not reimplement windowing in-chat** — run the script.

```bash
conda run -n caduceus_env python src/preprocessing.py \
  --raw raw \
  --out data_ready \
  --flank 10000 \
  --seed 42
```

| Flag | Default | Notes |
|------|---------|-------|
| `--raw` | `raw` | Root with `fna/`, `gtf/`, `tpm/` |
| `--out` | `data_ready` | Output directory |
| `--flank` | `10000` | Upstream/downstream flank (LOCKED) |
| `--seed` | `42` | Deterministic non-coding placement |
| `--genomes` | all | Optional GCF filter |
| `--max-genes` | none | Smoke-test cap |

Heavy panels → `sbatch src/sbatch/preprocess_raw.sbatch` (16 CPUs, 128G, 48h; runs `src/preprocessing.py`).

## Workflow checklist

```
adapt:
- [ ] Stage 1: Audit raw/{fna,gtf,tpm} + mapping; skip incomplete; abort if empty
- [ ] Stage 2: Confirm docs/caduceus_format.md + wiki/conversion.md
- [ ] Stage 3: CDS±10kb; neighbour trim; large-gene crop; non-coding match
- [ ] Write data_ready/* + caduceus_ready/*
- [ ] Update method-decision.md + artifact-registry + wiki/conversion.md
```

## Additional resources

- [wiki/conversion.md](../../wiki/conversion.md) — algo + I/O structures
- [README.md](README.md)
- Project entry: [`src/preprocessing.py`](../../src/preprocessing.py)
