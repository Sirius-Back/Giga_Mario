---
name: alpha-diversity
description: >-
  Compute alpha diversity (Observed, Shannon, Simpson, InvSimpson; Faith's PD
  if tree present) and plot by all target variables as boxplots or rainclouds
  (ggviolinbox: halfviolin + halfboxplot + jitter) with p-values. Prefers
  rarefied phyloseq when available. Use when the user mentions alpha-diversity,
  Shannon, raincloud, ggviolinbox, or alpha boxplots.
disable-model-invocation: true
---

# Alpha Diversity

## Purpose

Need to use rarefied object if it have been called before, or the raw object if not.
Output: boxplots OR rainclouds (ggviolinbox: halfviolin + halfboxplot + jitter) with p-values

Compare alpha diversity across **all target variables** (RDS `$target` or `--targets`).

Follow: **validation-first**, **reproducibility**, **statistical-analysis**, **publication-figures**, **method-decision-tracking**.

## Input resolution

1. `--rds` if given
2. Else prefer rarefied artifacts beside count RDS:
   - `*_rare.rds` / `grazing_phyloseq_rare.rds`
   - or `test/rarefaction-analysis/**/phyloseq_rare_*.rds`
3. Else raw count phyloseq (`grazing_phyloseq.rds` / `--rds`)

Do **not** use MMUPHin relative (`*_batchadj.rds`) for alpha unless counts.

## Targets

- Default: RDS list `$target` (column name(s))
- Override: `--targets grazing,Condition` (comma-separated)
- One figure set per target variable

## Plot style

| `--style` | Layers |
|-----------|--------|
| `raincloud` (default) | `geom_halfviolin` + `geom_halfboxplot` + `geom_jitter` |
| `boxplot` | `geom_boxplot` + `geom_jitter` |

P-values: Kruskal–Wallis overall + pairwise Wilcoxon (BH) via `ggpubr::stat_compare_means` / `rstatix` (omit NS brackets when `hide.ns`).

## Workflow

```
Alpha-diversity:
- [ ] Step 1: Resolve rarefied RDS if present, else raw counts
- [ ] Step 2: Resolve all target variables; validate ≥2 levels each
- [ ] Step 3: estimate_richness (+ Faith's PD if tree)
- [ ] Step 4: Stats table (KW + pairwise BH) per target × measure
- [ ] Step 5: Plot raincloud OR boxplot with p-values; save PDF/PNG
- [ ] Step 6: Write alpha-diversity-report.json
```

## Executable

```bash
Rscript .cursor/skills/alpha-diversity/scripts/alpha_diversity.R \
  --outdir test/alpha-diversity/grazing \
  --style raincloud

Rscript .cursor/skills/alpha-diversity/scripts/alpha_diversity.R --self-test
```

Optional: `--rds PATH` `--targets grazing` `--style boxplot` `--measures Observed,Shannon,Simpson,InvSimpson`

Thin hook: `.cursor/hooks/alpha-diversity.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `alpha_long.tsv` | Sample × measure × targets |
| `alpha_stats.tsv` | KW + pairwise p-values |
| `alpha_<target>_raincloud.pdf/.png` or `…_boxplot…` | Faceted plots with p-values |
| `alpha-diversity-report.json` | Input, targets, measures, paths |
