---
name: ancombc
description: >-
  Run ANCOM-BC2 differential abundance on rarefied phyloseq across all or
  specified targets and taxonomic aggregation levels (ASV + tax_level ranks).
  Use when the user mentions ancombc, ANCOM-BC, ANCOMBC, differential abundance,
  or tax_level DA tables for heat trees.
disable-model-invocation: true
---

# ANCOM-BC2

## Purpose

Need to be based on the RAREFIED phyloseq object. Target variables: use all OR specified
Better run `ancombc` on the any aggragation level (check the results to ensure this - it will be needed on the next stage

Follow: **validation-first**, **reproducibility**, **statistical-analysis**, **method-decision-tracking**, **slurm-execution-policy**, **artifact-registry**.

## Input resolution

1. `--rds` if given (must be **rarefied** count-like unless `--allow-non-rare true`)
2. Else require rarefied: `*_rare.rds` / `grazing_phyloseq_rare.rds` / `test/rarefaction-analysis/**/phyloseq_rare_*.rds`

Do **not** use MMUPHin relative (`*_batchadj.rds`).

## Targets

- Default: RDS `$target` (all listed) or discover categorical columns (2–12 levels)
- Override: `--targets grazing,Condition`
- One ANCOM-BC2 model per target (`fix_formula` / `group` = target)

## Aggregation levels

Default: **ASV** (`tax_level = NULL`) **plus every** `rank_names()` except `tip_rank`.

Override: `--levels ASV,Phylum,Family,Genus`

Uses `ANCOMBC::ancombc2(..., tax_level = <rank>)` (not a separate `tax_glom` pass). Results at each level are written for downstream heat-tree / top-taxon tables.

## Model defaults

| Parameter | Default |
|-----------|---------|
| Method | `ancombc2` |
| `p_adj_method` | `fdr` |
| `prv_cut` | `0.1` |
| `pseudo_sens` | `false` (enable with `--pseudo-sens true`) |
| `pairwise` | `true` when target has >2 levels |
| `struc_zero` / `neg_lb` | `false` |

Optional: `--covariates batch` → `fix_formula = "target + batch"`.

## Workflow

```
ANCOM-BC:
- [ ] Step 1: Resolve rarefied phyloseq; fail if non-rare (unless allowed)
- [ ] Step 2: Resolve targets (all or --targets)
- [ ] Step 3: Resolve aggregation levels (ASV + ranks, or --levels)
- [ ] Step 4: ancombc2 per target × level; tidy lfc/se/p/q tables
- [ ] Step 5: Write per-level TSV + combined results + ancombc-report.json
```

## Executable

```bash
Rscript .cursor/skills/ancombc/scripts/ancombc.R \
  --outdir test/ancombc/grazing

Rscript .cursor/skills/ancombc/scripts/ancombc.R --self-test

# Full multi-level job on SLURM
sbatch .cursor/skills/ancombc/scripts/ancombc.sbatch
```

Optional: `--rds PATH` `--targets grazing` `--levels ASV,Family,Genus` `--prv-cut 0.1` `--covariates batch`

Thin hook: `.cursor/hooks/ancombc.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `ancombc_results.tsv` | Long table: target, level, taxon, term, lfc, se, W, p, q |
| `ancombc_<target>_<level>.tsv` | Wide `ancombc2` `res` per target × level |
| `ancombc_results.rds` | Nested list `[target][[level]]` of res data.frames |
| `ancombc-report.json` | Input, rarefied depth, targets, levels, n_sig, paths |
