---
name: difftree-metacoder
description: >-
  Differential or default metacoder heat trees from rarefied phyloseq. Imports
  existing ancombc results when present (never re-runs ancombc); otherwise draws
  default abundance heat trees. Use when the user mentions difftree-metacoder,
  differential heat tree, ANCOM metacoder tree, or LFC heat_tree.
disable-model-invocation: true
---

# Diff Tree (metacoder)

## Purpose

Need to be based on the RAREFIED phyloseq object. If `ancombc` have been previously done, need to import its data; if not, use default metacoder. Test with both conditions (default, ancombc). Do not rerun ancombc if it is ok. Render the plots for it

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **artifact-registry**.

## Modes

| Mode | When | Node color |
|------|------|------------|
| `ancombc` | Existing ANCOM-BC tables found / `--mode ancombc` | log2 LFC (diverging) |
| `default` | No ANCOM-BC / `--mode default` | Mean abundance (`total`) |
| `auto` (default) | Prefer ancombc if OK report/CSV exists | as above |

**Never** call or re-run the `ancombc` skill when results are OK.

## Input resolution

### Phyloseq / Taxmap (rarefied required)

1. `--rds` rarefied phyloseq (or list with `$phyloseq`)
2. Else `grazing_phyloseq_rare.rds` / `test/rarefaction-analysis/**/phyloseq_rare_*.rds`
3. Optional `--metacoder` Taxmap; else `parse_phyloseq` from rarefied object

### ANCOM-BC import (no re-run)

Look for (first hit wins):

1. `--ancombc-csv` or `--ancombc-dir`
2. `test/ancombc/grazing/ancombc_results.csv` (or `ancombc2_results_all_levels.csv`)
3. `test/ancombc/**/ancombc_results.csv`
4. Fallback: `ancombc_results.tsv` / nested `ancombc_results.rds`

OK check: `ancombc-report.json` with all `level_summary[].ok == true`, or non-empty results table. If report says failed levels → **stop** (do not re-run).

## ANCOM path (codebase `ver3` / Kristina)

1. `calc_taxon_abund` → `leaf`
2. Map long ANCOM rows (all levels) → `taxon_id` (ASV via `otu_id`; ranks via name+rank)
3. Per target × term: pick min-p effect; `log2_lfc = lfc / log(2)`; zero if p > `--p-cut` (default 1)
4. `filter_taxa(leaf >= min_leaf & abs(log2_lfc) > lfc_cut)` (default leaf 1, lfc_cut 0.5)
5. `heat_tree(node_color = log2_lfc, node_color_range = green–gray–magenta)`

## Default path

Family merge (like `heattree`) + `heat_tree(node_color = total)`.

## Workflow

```
Difftree-metacoder:
- [ ] Step 1: Resolve rarefied phyloseq → Taxmap
- [ ] Step 2: Detect / import ancombc (never re-run)
- [ ] Step 3: Mode ancombc → attach LFC; mode default → abundance
- [ ] Step 4: Render PDF/PNG heat trees
- [ ] Step 5: Write difftree-metacoder-report.json
```

## Executable

```bash
# Auto (uses ancombc if present)
Rscript .cursor/skills/difftree-metacoder/scripts/difftree_metacoder.R \
  --outdir test/difftree-metacoder/grazing

# Force default (ignore ancombc)
Rscript .cursor/skills/difftree-metacoder/scripts/difftree_metacoder.R \
  --mode default --outdir test/difftree-metacoder/grazing-default

# Force ancombc import
Rscript .cursor/skills/difftree-metacoder/scripts/difftree_metacoder.R \
  --mode ancombc --ancombc-dir test/ancombc/grazing \
  --outdir test/difftree-metacoder/grazing-ancombc

Rscript .cursor/skills/difftree-metacoder/scripts/difftree_metacoder.R --self-test
```

Thin hook: `.cursor/hooks/difftree-metacoder.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `difftree_<target>_<term>.pdf/.png` | LFC heat trees (ancombc mode) |
| `difftree_default.pdf/.png` | Abundance heat tree (default mode) |
| `difftree_taxmap.rds` | Taxmap with attached columns |
| `difftree-metacoder-report.json` | Mode, inputs, figure paths |
