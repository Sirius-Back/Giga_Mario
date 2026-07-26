---
name: beta-diversity
description: >-
  Compute beta diversity PCoA for all target variables using Aitchison and
  weighted UniFrac by default (other distances optional), run PERMANOVA
  (adonis2), and annotate p/R² on plots. Prefers rarefied phyloseq when
  available. Use when the user mentions beta-diversity, PCoA, Aitchison,
  wUniFrac, PERMANOVA, or adonis.
disable-model-invocation: true
---

# Beta Diversity

## Purpose

Need to use rarefied object if it have been called before, or the raw object if not.
MUST use by default Aitchison & WUNIFRAC distances, but can be specified; AND calcualate PERMANOVA & show its results on the plot

Compare community structure across **all target variables** (RDS `$target` or `--targets`).

Follow: **validation-first**, **reproducibility**, **statistical-analysis**, **publication-figures**, **method-decision-tracking**.

## Input resolution

1. `--rds` if given
2. Else prefer rarefied: `*_rare.rds` / `grazing_phyloseq_rare.rds` / `test/rarefaction-analysis/**/phyloseq_rare_*.rds`
3. Else raw count phyloseq

Do **not** use MMUPHin relative (`*_batchadj.rds`) unless counts.

## Distances

| Name | Meaning | Default |
|------|---------|---------|
| `aitchison` | CLR + Euclidean (relative → CLR) | **yes** |
| `wunifrac` | Weighted UniFrac (needs tree) | **yes** |
| `bray` | Bray–Curtis | optional |
| `unifrac` | Unweighted UniFrac | optional |
| `jaccard` | Jaccard | optional |

Override: `--distances aitchison,wunifrac,bray`

## PERMANOVA

- `vegan::adonis2(dist ~ target, permutations = 999)` per distance × target
- Annotate each PCoA panel with **p** and **R²** (grazing `geom_richtext` pattern)

## Workflow

```
Beta-diversity:
- [ ] Step 1: Resolve rarefied RDS if present, else raw counts
- [ ] Step 2: Resolve all target variables; validate ≥2 levels
- [ ] Step 3: Build relative (+ CLR for Aitchison); compute distances
- [ ] Step 4: PCoA + PERMANOVA per distance × target
- [ ] Step 5: Plot points + 95% ellipses + PERMANOVA annotation
- [ ] Step 6: Write beta-diversity-report.json + stats TSV
```

## Executable

```bash
Rscript .cursor/skills/beta-diversity/scripts/beta_diversity.R \
  --outdir test/beta-diversity/grazing

Rscript .cursor/skills/beta-diversity/scripts/beta_diversity.R --self-test
```

Optional: `--rds PATH` `--targets grazing` `--distances aitchison,wunifrac` `--permutations 999`

Thin hook: `.cursor/hooks/beta-diversity.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `beta_permanova.tsv` | PERMANOVA p / R² per target × distance |
| `beta_<target>_pcoa.pdf/.png` | Faceted PCoA with PERMANOVA labels |
| `beta-diversity-report.json` | Input, distances, targets, paths |
