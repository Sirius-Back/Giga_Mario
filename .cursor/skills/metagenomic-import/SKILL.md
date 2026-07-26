---
name: metagenomic-import
description: >-
  Import WGS/metagenomic Bracken reports into a complete phyloseq object (tax,
  otu, sam, tree) with host cleanup. Use when importing Bracken/Kraken reports,
  building WGS phyloseq, or when the user mentions metagenomic-import.
disable-model-invocation: true
---

# Metagenomic Import

## Purpose

Build a **complete** phyloseq object from WGS Bracken / Kraken-style reports.

Hooks MUST use: metadata; data (all available); than reconstruct the tree. If there is missing metadata -> exit; else you can use skill @fix-metadata
report the structure of the final phyloseq object. it MUST have:
* tax table
* otu table
* sam data
* tree data (novel or found)

Follow project rules: **validation-first**, **reproducibility**, **missing-data-policy**, **method-decision-tracking**. Respect Locked decisions in `method-decision.md`.

## Workflow

```
Metagenomic import:
- [ ] Step 1: Resolve indir / metadata (run @fix-metadata if missing/misaligned)
- [ ] Step 2: Discover all Bracken/report files
- [ ] Step 3: Require metadata — exit if missing
- [ ] Step 4: Parse all available abundance files → host cleanup → finalize taxonomy → phyloseq
- [ ] Step 5: Reconstruct tree (taxonomy-tree ≤40 taxids else formula/hclust)
- [ ] Step 6: Assert + report phyloseq structure (tax, otu, sam, tree)
- [ ] Step 7: Register outputs in artifact-registry.md
```

### Step 1 — Metadata first

- If metadata is missing → **exit** and invoke `@fix-metadata`.
- If metadata/sample map do not align with Bracken sample IDs → **exit** and invoke `@fix-metadata`.
- Prefer `metadata_fixed.csv` from `@fix-metadata` when present.
- Real honey layout: pass `--metadata …/legends/sra.csv` or run `@fix-metadata` on `data/` / `k2/`.

### Step 4 — Data

Parse **all available** `*.nt.G.bracken` / `*.nt.bracken.[SG].report` under `indir` (optional `--max-files` only for smoke tests). Apply Locked host/Chordata cleanup.

### Step 5 — Tree

Tree is obligatory. Small taxid sets → `taxonomy-tree` hook; large → formula/`hclust` (Locked).

## Executable

```bash
Rscript .cursor/skills/metagenomic-import/scripts/metagenomic_import.R \
  --indir path/to/wgs --outdir test/metagenomic-import/run \
  --metadata path/to/sra.csv

Rscript .cursor/skills/metagenomic-import/scripts/metagenomic_import.R --self-test
```

Thin hook wrapper (compat): `.cursor/hooks/metagenomic-import.sh`

## Outputs

| Artifact | Path |
|----------|------|
| Phyloseq | `{outdir}/phyloseq.rds` |
| Bracken parse | `{outdir}/bracken_parsed.rds` |
| Report | `{outdir}/metagenomic-import-report.json` |

## Shared code

- `.cursor/skills/_shared/import/import_common.R`
- `.cursor/skills/_shared/import/bracken_parse.R`
