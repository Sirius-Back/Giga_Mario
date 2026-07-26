---
name: metabarcoding-import
description: >-
  Import 16S/metabarcoding data into a complete phyloseq object (tax, otu, sam,
  tree). Use when importing QIIME2 qza or feature-table/taxonomy TSV, building
  phyloseq from 16S inputs, or when the user mentions metabarcoding-import.
disable-model-invocation: true
---

# Metabarcoding Import

## Purpose

Build a **complete** phyloseq object from 16S / metabarcoding inputs.

Hooks MUST use: metadata; data (all available); than reconstruct the tree. If there is missing metadata -> exit; else you can use skill @fix-metadata
report the structure of the final phyloseq object. it MUST have:
* tax table
* otu table
* sam data
* tree data (novel or found)

Follow project rules: **validation-first**, **reproducibility**, **missing-data-policy**, **method-decision-tracking**. Respect Locked decisions in `method-decision.md`.

## Workflow

```
Metabarcoding import:
- [ ] Step 1: Resolve indir / metadata (run @fix-metadata if missing/misaligned)
- [ ] Step 2: Discover all available 16S artifacts
- [ ] Step 3: Require metadata — exit if missing
- [ ] Step 4: Import all available data → finalize taxonomy (no Unclassified tip; strip Candidatus_; ASV disambiguation) → phyloseq
- [ ] Step 5: Reconstruct tree if missing (taxonomy-tree / formula / nwk)
- [ ] Step 6: Assert + report phyloseq structure (tax, otu, sam, tree)
- [ ] Step 7: Register outputs in artifact-registry.md
```

### Step 1 — Metadata first

- If metadata is missing → **exit** and invoke `@fix-metadata`.
- If metadata exists but sample IDs do not overlap abundance columns → **exit** and invoke `@fix-metadata`.
- Prefer `metadata_fixed.csv` from `@fix-metadata` when present.

### Step 2–4 — Data

Prefer `qiime2R::qza_to_phyloseq` when `table.qza` is present (Locked). Otherwise plain TSV. Use **all available** co-located artifacts (taxonomy, sequences, tree).

### Step 5 — Tree

If tree missing → reconstruct (existing `.nwk`, `taxonomy-tree` for small sets, formula/`hclust` fallback). Import **fails** if tree cannot be attached.

### Step 6 — Report structure

Print and write `phyloseq_structure` in the JSON report. All four slots must be TRUE.

## Executable

```bash
Rscript .cursor/skills/metabarcoding-import/scripts/metabarcoding_import.R \
  --indir path/to/16s --outdir test/metabarcoding-import/run

Rscript .cursor/skills/metabarcoding-import/scripts/metabarcoding_import.R --self-test
```

Optional: `--metadata path/to/sample-metadata.tsv`

Thin hook wrapper (compat): `.cursor/hooks/metabarcoding-import.sh`

## Outputs

| Artifact | Path |
|----------|------|
| Phyloseq | `{outdir}/phyloseq.rds` |
| Report | `{outdir}/metabarcoding-import-report.json` |
| Discovery | `{outdir}/discovery.json` |

## Shared code

- `.cursor/skills/_shared/import/import_common.R`
