---
name: metabolism-de
description: >-
  Differential abundance of metabolic gene/function tables (Bakta product/KO/EC,
  never GO) with ANCOM-BC2. Volcano + top-20 LFC±SE barplots. Use for
  metabolism-DE, metabolic ANCOM, Bakta pathway DA, or functional DEG without GO.
disable-model-invocation: true
---

# Metabolism-DE (ANCOM-BC2 on metabolic tables)

## Purpose

Run **ANCOM-BC2** on Bakta-style metabolic abundance tables (`product` / `ko` / `ec`). **Never includes GO** (use `@go`).

Follow: **validation-first**, **statistical-analysis**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **artifact-registry**, **slurm-execution-policy**.

## Related codebase

| Source | Role |
|--------|------|
| Kristina `run_ancombc2_bakta_pathways.R` | ANCOM-BC2 on Bakta function phyloseq |
| `@metabolism` | Import / abundance heatmaps (no DE) |
| `@ancombc` | Taxonomic ANCOM-BC2 pattern (extract lfc/se/p/q) |

## Input resolution

1. `--long` Bakta long CSV/TSV
2. Else `--matrix` / `--rds` counts
3. Else metabolism / Kristina long candidates

**Metadata** required: `--metadata` or `--ps-rds` with `--group-col` (default `group`).

## Defaults

| Parameter | Default |
|-----------|---------|
| Types | `product` (`--types product,ko,ec`) |
| GO | always dropped |
| Method | `ANCOMBC::ancombc2` |
| `prv_cut` | `0.1` |
| `p_adj_method` | `fdr` |
| Plots | volcano + top-20 \|LFC\| ± SE |

## Executable

```bash
Rscript .cursor/skills/metabolism-de/scripts/metabolism_de.R \
  --long PATH/bakta_function_long.csv \
  --metadata PATH/metadata.csv \
  --group-col group \
  --outdir test/metabolism-de/run

Rscript .cursor/skills/metabolism-de/scripts/metabolism_de.R --self-test
```

Thin hook: `.cursor/hooks/metabolism-de.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `metabolism_de_results.tsv` | Long ANCOM-BC2 (feature, term, lfc, se, p, q, log2_lfc) |
| `metabolism_de_volcano.pdf/.png` | Volcano |
| `metabolism_de_top20_lfc.pdf/.png` | Top-20 LFC ± SE |
| `metabolism-de-report.json` | Run metadata |
