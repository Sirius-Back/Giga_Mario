---
name: ordination
description: >-
  Ordination by all target variables: default sPLS-DA (rarefied required; main
  plot with geom_rect prediction regions + two component-loading plots) or NMDS
  (ggplot2; batch & target envfit; top-N feature arrows). Prefers rarefied
  phyloseq when available. Use when the user mentions ordination, sPLS-DA,
  mixOmics, NMDS, or envfit.
disable-model-invocation: true
---

# Ordination

## Purpose

Need to use rarefied object if it have been called before, or the raw object if not.
Possible ways:
* `sPLS-DA` . Need to use rarefied object. Output: 3 ggplots: main with geom_rect to split the regions; 2 with component loadings. Default
* `NMDS`. Same. ggplot2-based. Also report batch & target envfit & draw the arrows for the main influencing features (top-5 by default)

Compare across **all target variables** (RDS `$target` or `--targets`).

Follow: **validation-first**, **reproducibility**, **statistical-analysis**, **publication-figures**, **method-decision-tracking**.

## Input resolution

1. `--rds` if given
2. Else prefer rarefied: `*_rare.rds` / `grazing_phyloseq_rare.rds` / `test/rarefaction-analysis/**/phyloseq_rare_*.rds`
3. Else raw counts (NMDS only)

| Method | Rarefied |
|--------|----------|
| `splsda` (default) | **Required** — exit if only raw available |
| `nmds` | Preferred; raw allowed |

## Methods

### sPLS-DA (`--method splsda`)

- `mixOmics::splsda` on rarefied OTU (samples × taxa), `ncomp = 2`, `keepX = c(10,10)` default
- **Plot 1 (main):** sample scores + `geom_rect` class prediction regions (grid + `predict`)
- **Plot 2:** loadings for component 1
- **Plot 3:** loadings for component 2

### NMDS (`--method nmds`)

- `vegan::metaMDS` (Bray–Curtis on relative abundances)
- `envfit` for **batch** and **target**; write `nmds_envfit.tsv`
- Arrows for top-N taxa by envfit r² (`--top-features 5` default)

## Workflow

```
Ordination:
- [ ] Step 1: Resolve rarefied (required for sPLS-DA) or raw
- [ ] Step 2: Resolve all target variables (+ batch for NMDS)
- [ ] Step 3: Fit sPLS-DA or NMDS
- [ ] Step 4: Save 3 sPLS-DA plots OR NMDS + envfit/arrows
- [ ] Step 5: Write ordination-report.json
```

## Executable

```bash
Rscript .cursor/skills/ordination/scripts/ordination.R \
  --outdir test/ordination/grazing \
  --method splsda

Rscript .cursor/skills/ordination/scripts/ordination.R \
  --outdir test/ordination/grazing-nmds \
  --method nmds --top-features 5

Rscript .cursor/skills/ordination/scripts/ordination.R --self-test
```

Optional: `--rds PATH` `--targets grazing` `--batch-var batch` `--keepX 10,10` `--seed 123`

Thin hook: `.cursor/hooks/ordination.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `splsda_<target>_main.pdf/.png` | Scores + geom_rect regions |
| `splsda_<target>_loading1/2.pdf/.png` | Component loadings |
| `nmds_<target>.pdf/.png` | NMDS + envfit arrows (batch/target + top features) |
| `nmds_envfit.tsv` | envfit r² / p for batch, target, features |
| `ordination-report.json` | Method, input, targets, paths |
