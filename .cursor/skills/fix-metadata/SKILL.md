---
name: fix-metadata
description: >-
  Find sample metadata and abundance/data files, check sample-ID alignment, and
  write metadata_fixed.csv when misaligned. Use when import hooks exit for
  missing or misaligned metadata, or when the user mentions fix-metadata.
disable-model-invocation: true
---

# Fix Metadata

## Purpose

Repair or normalize sample metadata so import skills can join abundances to samples.

* find metadata
* find all data, all its types
* check alignment of the data/metadata. using subagent, investigate metadata. if it is not aligned with data directly - copy & edit metadata_fixed.csv

Follow project rules: **validation-first**, **missing-data-policy**, **reproducibility**.

## Workflow

```
Fix metadata:
- [ ] Step 1: Find metadata candidates (indir + parents; sra.csv; sample-metadata; maps)
- [ ] Step 2: Find all data, all its types (Bracken, feature-table, reports)
- [ ] Step 3: Check alignment of data sample IDs vs metadata ID columns
- [ ] Step 4: If not aligned directly — investigate (subagent) + write metadata_fixed.csv
- [ ] Step 5: Report coverage; hand metadata_fixed.csv to import skills
```

### Step 4 — Subagent investigation

When automatic alignment fails or coverage is low:

1. Launch a `generalPurpose` or `explore` subagent with the data inventory JSON and candidate metadata paths.
2. Ask it to propose ID mappings (Run↔bracken basename, sample_map, stripped suffixes).
3. Apply the mapping into `metadata_fixed.csv` (copy & edit; never overwrite the original metadata file).

### Alignment strategies (script)

- Direct ID match
- Case-insensitive match
- Strip sequencing suffixes
- Prefer columns: `sampleID`, `Run`, `sample_id`, `bracken_id`

## Executable

```bash
Rscript .cursor/skills/fix-metadata/scripts/fix_metadata.R \
  --indir path/to/data --outdir test/fix-metadata/run

Rscript .cursor/skills/fix-metadata/scripts/fix_metadata.R --self-test
```

Optional: `--metadata path/to/candidate.csv`

## Outputs

| Artifact | Path |
|----------|------|
| Fixed metadata | `{outdir}/metadata_fixed.csv` |
| Inventory | `{outdir}/data-inventory.json` |
| Report | `{outdir}/fix-metadata-report.json` |

## Downstream

Pass `--metadata {outdir}/metadata_fixed.csv` to `@metabarcoding-import` or `@metagenomic-import`.
