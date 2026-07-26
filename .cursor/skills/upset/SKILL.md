---
name: upset
description: >-
  ComplexUpset plots of taxon presence/absence across groups. By default
  combines samples only by the target variable (MicrobiotaProcess::get_upset /
  ticks+PacBio ComplexUpset pattern). Use when the user mentions UpSet,
  ComplexUpset, set intersections, or taxon sharing across groups.
disable-model-invocation: true
---

# UpSet

## Purpose

Visualize taxon (ASV/OTU) set intersections across sample groups with **ComplexUpset**.

**Default:** combine samples **only by the target variable** — a taxon belongs to a target level if it is present (count > 0) in any sample of that level. Do **not** concatenate batch, host, geography, or other factors unless the user overrides `--factors`.

Source patterns: PacBio / Kristina `get_upset` + `ComplexUpset::upset`; ticks meta-analysis `ComplexUpset` when multi-column cohort tables exist. Details: [reference.md](reference.md).

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **artifact-registry**.

## Input resolution

1. `--rds` if given
2. Else prefer rarefied: `*_rare.rds` / `grazing_phyloseq_rare.rds` / `test/rarefaction-analysis/**/phyloseq_rare_*.rds`
3. Else raw counts

Presence is on counts (or any non-negative abundance); relative tables are allowed for presence only.

## Grouping (Locked default)

| Mode | Flag | Behavior |
|------|------|----------|
| **Target only (default)** | `--target grazing` or RDS `$target` | Sets = levels of one column |
| Multi-factor (opt-in) | `--factors genus,region` | Sets = unique combinations (`paste` like PacBio `full`) |

## Plot defaults (ComplexUpset)

| Parameter | Default |
|-----------|---------|
| `min_size` | `3` |
| `stripes` | `white` |
| `width_ratio` | `0.2` |
| `sort_intersections_by` | `degree`, `cardinality` |
| `sort_intersections` | `descending` (largest intersections left) |
| Set sizes left margin | +1.2 cm equivalent on taxa-sets panel |
| Intersection bar labels | **counts** (`--label-mode percent` for %) |
| Set queries | Colored per set (`upset_query`) |

## Workflow

```
UpSet:
- [ ] Step 1: Resolve phyloseq + target (or --factors)
- [ ] Step 2: Build presence matrix (get_upset / equivalent)
- [ ] Step 3: ComplexUpset::upset
- [ ] Step 4: Save PDF/PNG + upset matrix TSV + upset-report.json
```

## Executable

```bash
Rscript .cursor/skills/upset/scripts/upset.R \
  --outdir test/upset/grazing \
  --target grazing

Rscript .cursor/skills/upset/scripts/upset.R --self-test

sbatch .cursor/skills/upset/scripts/upset.sbatch
```

Optional: `--rds PATH` `--factors a,b` `--min-size 3` `--annotate COL` `--label-mode count|percent`

Thin hook: `.cursor/hooks/upset.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `upset_matrix.tsv` | Taxa × set 0/1 (+ optional annotation cols) |
| `upset.pdf` / `upset.png` | ComplexUpset figure |
| `upset-report.json` | Target/factors, n_sets, n_taxa, paths |

## Additional resources

- Codebase notes: [reference.md](reference.md)
