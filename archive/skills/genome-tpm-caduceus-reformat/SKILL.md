---
name: genome-tpm-caduceus-reformat
description: >-
  Pair RefSeq genomic FASTA/GTF with per-assay TPM (and optional genes.tsv) into
  sample manifests and Caduceus-oriented fold-ready layouts for multi-species
  random splits. Use when adapting random/ + data/raw/genomes panels for @split
  outputs under data_splits/full with coherent genome+transcriptome samples.
disable-model-invocation: true
---

# Genome + TPM Caduceus reformat

## Purpose

Build **coherent genome + transcriptome** sample tables from:

- RefSeq assemblies under `data/raw/genomes/<GCF>/` (`*_genomic.fna`, `*genomic.gtf*`)
- TPM matrices under `random/expression_data/<assay>/tpm/`
- Optional gene tables under `random/genes/<GCF>/`
- Pairing file `random/random_borzoi_expr_file_mappings.csv`

Does **not** assign train/val/test folds — that is `@split` + `splits/*.md`.

Follow: **validation-first**, **missing-data-policy**, **reproducibility**,
**method-decision-tracking**, **artifact-registry**, **task-status**.

## Required inputs

| Input | Meaning |
|-------|---------|
| **MAPPINGS** | CSV with columns including `id` (assay), `genome` (GCF) |
| **GENOMES_ROOT** | `data/raw/genomes` |
| **EXPR_ROOT** | `random/expression_data` |
| **OUT** | Project-relative reformat output (e.g. `data/reformat/random_full`) |

Optional: `GENES_ROOT=random/genes`, `require_tpm=true` (default), `exclude_accessions=…`.

## Pair discovery

1. Read mappings; one row per assay↔GCF.
2. For each row resolve:
   - `fna_path` = first `*_genomic.fna` under `GENOMES_ROOT/<GCF>/`
   - `gtf_path` = first `*genomic.gtf` or `*genomic.gtf.gz`
   - `tpm_path` = `EXPR_ROOT/<id>/tpm/<id>.csv`
   - `genes_path` = `GENES_ROOT/<GCF>/<GCF>_genes.tsv` if present
3. If `require_tpm=true` and TPM missing → **exclude** sample; record in `exclusions.tsv` (do not invent TPM).
4. Fail early if FNA or GTF missing for a non-excluded sample.
5. Species label = NCBI organism name when known, else GCF accession.

## Caduceus-oriented layout (fold-ready assets, not fold membership)

Upstream Caduceus `GenomicBenchmarkDataset` expects:

```
{dest}/{dataset_name}/{split}/{label}/*.txt   # raw DNA (no FASTA header)
```

Whole mammalian assemblies are **not** single GB examples. This skill prepares:

| Output under `OUT` | Content |
|--------------------|---------|
| `manifest.tsv` | `sample_id,species,genome_accession,assay_id,fna_path,gtf_path,tpm_path,genes_path` (relative paths) |
| `exclusions.tsv` | Excluded rows + reason |
| `selection.json` | `{seed_policy, n_included, n_excluded, require_tpm}` |
| `caduceus_notes.md` | How folds should place files for Caduceus custom use |

After `@split` assigns folds, each fold directory should contain **per sample**:

```
{OUT_FOLDS}/{train|val|test}/{sample_id}/
  genome.fna          # hardlink/symlink to genomic FASTA
  annotation.gtf      # or .gtf.gz
  expression_tpm.csv  # TPM matrix
  genes.tsv           # optional
```

Optional GB smoke tree (only if user asks): extract short gene windows to
`{fold}/{sample_id}/sequences/*.txt` (raw ACGT). Default: **off** (too large for full geneomes).

## Workflow

```
genome-tpm-caduceus-reformat:
- [ ] Validate mappings + roots exist
- [ ] Resolve paths; exclude missing TPM when require_tpm
- [ ] Write manifest.tsv + exclusions.tsv + selection.json + caduceus_notes.md
- [ ] Record method-decision.md
- [ ] Register artifacts
- [ ] Hand off to @split (no fold assignment here)
```

## Coordination

| Skill | Role |
|-------|------|
| `@split` | Species-level random fold assignment → `data_splits/full/` |
| `@genome-fna-gtf-reformat` | FNA/GTF-only panels (no TPM) |
| `@caduceus` | Training after folds exist |
| `@get-data` / `@data` | Acquire missing FASTA/TPM upstream |
