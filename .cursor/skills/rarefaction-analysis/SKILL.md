---
name: rarefaction-analysis
description: >-
  Build rarefaction curves and rarefy phyloseq to even depth (default: lowest
  depth if ≥1000 reads, else 1000 after dropping shallow samples). Prefers
  batch-adjusted count objects when available. Use when the user mentions
  rarefaction-analysis, rarefy, rarefaction curves, or even-depth sampling.
disable-model-invocation: true
---

# Rarefaction Analysis

## Purpose

If batch have been removed, use this object; if not, use directly
* create rarefaction curves
* rarefy at the at least 1000 reads OR the lowest sequencing depth IF NOT specified in the other way
* rarefy only here; if different rarefied sequencing depth is needed, save several objects

Rarefaction is for alpha-diversity / even-depth objects only (Kristina / grazing pattern). Do not use rarefied tables for compositional DA unless explicitly requested.

Follow: **validation-first**, **reproducibility**, **method-decision-tracking**, **publication-figures**.

## Input resolution

1. `--rds` if given
2. Else if `prefer_batchadj` (default): try `*_batchadj.rds` beside a count RDS
3. **Counts required:** if `min(sample_sums) ≈ 1` (relative / MMUPHin output), fall back to the non-batchadj count RDS and record the fallback in the report

## Depth rule (when `--depth` / `--depths` omitted)

| Condition | Depth |
|-----------|--------|
| `min(sample_sums) ≥ 1000` | `min(sample_sums)` (lowest sequencing depth) |
| `min(sample_sums) < 1000` | `1000`, after pruning samples with sum `< 1000` |

Multiple depths: `--depths 1000,2150` → one rarefied RDS per depth (`phyloseq_rare_<depth>.rds`).

## Workflow

```
Rarefaction-analysis:
- [ ] Step 1: Resolve RDS (batchadj if counts; else raw)
- [ ] Step 2: Validate integer-like counts; compute depth(s)
- [ ] Step 3: Alpha rarefaction curves (Observed, Shannon, Simpson) → PNG/PDF
        — exact samples: geom_line; target: geom_smooth
- [ ] Step 4: rarefy_even_depth per depth; save separate objects
- [ ] Step 5: Structure report (tax/otu/sam/tree) per object
```

## Executable

```bash
Rscript .cursor/skills/rarefaction-analysis/scripts/rarefaction_analysis.R \
  --rds test/code-review-phyloseq/grazing_phyloseq.rds \
  --outdir test/rarefaction-analysis/grazing

Rscript .cursor/skills/rarefaction-analysis/scripts/rarefaction_analysis.R --self-test
```

Optional: `--depths 1000,2000` `--seed 123` `--n_reps 5` `--color_by grazing`

Thin hook: `.cursor/hooks/rarefaction-analysis.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `rarefaction_curves.pdf` / `.png` | Faceted Observed/Shannon/Simpson: per-sample lines + `geom_smooth` by target |
| `rarefaction_curves.tsv` | Sample × Measure × Depth means |
| `rarefaction_alpha_long.tsv` | All rarefaction replicates |
| `phyloseq_rare_<depth>.rds` | One even-depth phyloseq per depth |
| `rarefaction-report.json` | Depths, samples dropped, structure per object |
