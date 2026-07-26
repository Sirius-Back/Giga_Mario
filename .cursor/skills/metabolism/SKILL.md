---
name: metabolism
description: >-
  Import WGS metabolic/functional gene tables (Bakta EC/KO/product; optional
  wide matrices) excluding GO, and plot the most abundant genes with pheatmap.
  Use when the user mentions metabolism, Bakta functions, KO/EC gene abundance,
  metabolic heatmap, or functional gene tables (not GO enrichment).
disable-model-invocation: true
---

# Metabolism (gene abundance tables + pheatmap)

## Purpose

Import metabolic annotation abundance as tables and visualize the **most abundant genes** with `pheatmap`.

**Out of scope:** Gene Ontology enrichment / GO plots — that is a separate skill. This skill **drops `go` rows** from Bakta-style tables and never runs GO analyses.

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **artifact-registry**.

## Related codebase patterns

| Source | Role |
|--------|------|
| Kristina `R/bakta_gff3.R` / `run_bakta_function_import.R` | GFF3 → long `sample,function_type,function_id,count` (+ optional matrix RDS) |
| Kristina `data/processed/bakta_function_long.csv` | Ready long table (product / ec / ko / go) |
| Kristina `ver3.Rmd` / `dsmutin.Rmd` | `pheatmap` green–white–magenta top-N abundance pattern (taxonomy; reuse palette) |
| `ixodes/WGS/metabolic.Rmd` | eggNOG gene-model × sample wide matrix (alternate import) |
| Resistance / AMR heatmaps | Presence `pheatmap` only — not used here |

No PICRUSt / HUMAnN / Tax4Fun / FAPROTAX pipelines in the mapped codebases.

## Input resolution

1. `--long` Bakta-style long CSV/TSV (`sample`, `function_type`, `function_id`, `count`)
2. Else `--matrix` / `--rds` wide counts (genes × samples) or Bakta list RDS with `$counts`
3. Else `--gff3-dir` (+ optional `--sample-map`) → import via Kristina `bakta_gff3.R` when present
4. Else `.cursor/skills/metabolism/fixtures/bakta_function_long.csv`, `test/metabolism/**`, or Kristina processed long CSV if reachable

## Defaults

| Parameter | Default |
|-----------|---------|
| Function types kept | `product` (override `--types product,ko,ec`) |
| GO | **always excluded** |
| Top genes | `--top-n 200` by mean relative abundance (`0` = all genes) |
| Drop hypothetical products | `--drop-hypothetical true` |
| Matrix values for heatmap | relative; **default** rarefy/normalize via related phyloseq `--ps-rds` / auto-discover (`--no-ps-rds` to skip) |
| `pheatmap` scale | `row` |
| Palette | `#1B9E77` → white → `#D81B60` (Kristina) |

## Workflow

```
Metabolism:
- [ ] Step 1: Resolve long table / matrix / GFF3; validate columns
- [ ] Step 2: Drop GO; keep --types; build counts + relative matrices
- [ ] Step 3: Select top-N genes; write tables
- [ ] Step 4: pheatmap PDF + PNG
- [ ] Step 5: Write metabolism-report.json
```

## Executable

```bash
Rscript .cursor/skills/metabolism/scripts/metabolism.R \
  --long PATH/bakta_function_long.csv \
  --outdir test/metabolism/run

Rscript .cursor/skills/metabolism/scripts/metabolism.R --self-test
```

Optional: `--types product,ko` `--top-n 50` `--scale none` `--gff3-dir DIR` `--sample-map CSV` `--matrix TSV` `--rds bakta_function_matrix.rds` `--drop-hypothetical false`

Thin hook: `.cursor/hooks/metabolism.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `metabolism_long.csv` | Filtered long table (no GO) |
| `metabolism_counts.tsv` | Genes × samples counts |
| `metabolism_rel.tsv` | Genes × samples relative abundances |
| `metabolism_top_genes.tsv` | Ranked top-N mean relative abundance |
| `metabolism_heatmap.pdf` / `.png` | Top-N gene pheatmap |
| `metabolism-report.json` | Inputs, types, n_genes, figure paths |
