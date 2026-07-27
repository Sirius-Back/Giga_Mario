---
name: summarize-GEO
description: >-
  Mean-merge GEO-aligned wide TPM CSVs per genome assembly and assess
  gene↔transcript linkage via GTF. Use when the user asks to summarize GEO TPM
  replicates, write assembly_merged.csv, or run summarize_GEO / summarize-geo.
disable-model-invocation: true
---

# Summarize GEO

## Purpose

Summarize per-sample GEO-aligned **wide TPM** CSVs into one mean-expression matrix per assembly, and verify that gene→transcript conversion from the matching GTF is direct (1:1).

Canonical code: [`src/summarize_geo.py`](../../src/summarize_geo.py).

Follow project rules: **validation-first**, **reproducibility**, **scientific-integrity**, **artifact-registry**.

## Inputs

| Input | Typical path |
|-------|----------------|
| Per-sample TPM (wide) | `prokaryotes/tpm/GSE*_*.csv` — header=`gene_id`, one TPM row |
| Mappings | `prokaryotes/expr_file_mappings.csv` (`genome_stem`, `tpm`, …) |
| GTF (optional assess) | `prokaryotes/gtf/{assembly}_genomic.gtf` |

## Outputs

| Output | Path |
|--------|------|
| Mean TPM per assembly | `prokaryotes/tpm/{assembly}_merged.csv` |
| Optional transcript assess | stdout table from `--assess-transcripts` |

`{assembly}` = `genome_stem` (e.g. `GCF_000005845.2_ASM584v2`).

## Workflow

```
Summarize GEO:
- [ ] Step 1: Validate mappings + TPM paths exist and are non-empty
- [ ] Step 2: Group samples by genome_stem
- [ ] Step 3: Mean-merge (intersection genes; renormalize ΣTPM=1e6)
- [ ] Step 4: Write {assembly}_merged.csv
- [ ] Step 5: (Optional) assess gene→transcript ease on GTFs
- [ ] Step 6: Register artifacts
```

### Step 1–4 — Mean merge

```bash
python src/summarize_geo.py \
  --mappings prokaryotes/expr_file_mappings.csv \
  --tpm-dir prokaryotes/tpm \
  --prok-root prokaryotes
```

### Step 5 — Gene→transcript ease

```bash
python src/summarize_geo.py --assess-transcripts --prok-root prokaryotes
```

For RefSeq prokaryotes, expression keys are **gene_id**; CDS rows usually carry a single `transcript_id` (`unassigned_transcript_*` or one RNA transcript). Treat gene_id as the training/expression key unless multi-transcript genes are flagged.

## Core functions (`src/summarize_geo.py`)

| Function | Role |
|----------|------|
| `read_wide_tpm` / `write_wide_tpm` | I/O for wide TPM CSVs |
| `mean_merge_tpm_dicts` / `mean_merge_tpm_csvs` | Element-wise mean (+ optional renormalize) |
| `load_assembly_tpm_groups` | Group paths by `genome_stem` from mappings |
| `summarize_assemblies` | Write all `{assembly}_merged.csv` |
| `parse_gtf_gene_transcript_map` | gene_id → coords + transcript_ids |
| `gene_to_transcript_easy` | 1:1 map; raises if multi-transcript genes exist |
| `assess_panel_gene_transcript` | Panel-wide ease report |

## Rules

- Do **not** invent expression values; only average existing TPM columns.
- Default gene set = **intersection** across the assembly’s sample CSVs.
- After mean, **renormalize** so ΣTPM = 1e6 (disable with `--no-renormalize`).
- Fail early if mappings or TPM files are missing/empty.
- Record method choice in `method-decision.md` when defaults change.

## Artifact registration

Register every generated `{assembly}_merged.csv` set and any new report in `docs/artifact-registry.md` (producer skill: `summarize-GEO`).
