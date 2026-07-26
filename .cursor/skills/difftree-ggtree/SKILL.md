---
name: difftree-ggtree
description: >-
  Differential ggtree from rarefied phyloseq + required prior ancombc (multilevel).
  Default PacBio-style cladobox: circular tree with highlighted DA taxa + side
  abundance boxplots. Also fruit circular/rectangular and twosided. Never re-runs
  ancombc. Use when the user mentions difftree-ggtree, ggdiffclade, ggtreeExtra,
  or ANCOM ggtree.
disable-model-invocation: true
---

# Diff Tree (ggtree)

## Purpose

Rarefied phyloseq + previous **`ancombc`** (never re-run). Default visual matches PacBio `ggdiffclade` + `ggdiffbox` (`ggarrange` widths 0.7/1), driven by **multilevel ANCOM-BC** (Genus → Family → … → ASV).

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **artifact-registry**.

## Requirements

| Input | Rule |
|-------|------|
| Phyloseq | **Rarefied** counts |
| ANCOM-BC | **Required** multilevel OK results; **never re-run** |
| Tree | `phy_tree` preferred; else taxonomy formula tree |

## Layouts

| `--layout` | Description |
|------------|-------------|
| **`cladobox` (default)** | Circular tree, tips sized by −log10(p) / colored by log2 LFC, + abundance boxplots (`ggarrange` 0.7∶1) |
| `twosided` | Rectangular tree + LFC strip + boxes |
| `circular` / `fruit` | Grazing `geom_fruit` LFC bars (tree → tips → fruit) |
| `rectangular` | Same fruit pattern, rectangular |

## Multilevel

Default for `cladobox`: plot each available preferred level among `Genus,Family,Order,Class,Phylum,ASV`.

**Tree source (Locked):** always the phyloseq object — `phy_tree` preferred, else taxonomy formula from `tax_table`. Synthetic/index-order trees are **forbidden**. Tips absent from that tree are dropped; &lt;2 tips → fail.

Override: `--levels Genus,ASV`

## Executable

```bash
Rscript .cursor/skills/difftree-ggtree/scripts/difftree_ggtree.R \
  --outdir test/difftree-ggtree/grazing \
  --ancombc-dir test/ancombc/grazing

Rscript .cursor/skills/difftree-ggtree/scripts/difftree_ggtree.R \
  --layout circular --tip-offset 0.5 --fruit-offset 0

Rscript .cursor/skills/difftree-ggtree/scripts/difftree_ggtree.R --self-test
```

Optional: `--rds` `--layout cladobox|twosided|circular|rectangular` `--levels Genus,ASV` `--max-tips 80` `--lfc-cut 0.5` `--q-cut 0.05`

## Outputs

| Artifact | Content |
|----------|---------|
| `difftree_ggtree_<target>_<term>_<level>.pdf/.png` | Cladobox / fruit / twosided |
| `difftree_ggtree_tips.tsv` | Tip selection |
| `difftree-ggtree-report.json` | Inputs, levels, figures |
