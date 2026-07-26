---
name: phyloseq2metacoder
description: >-
  Convert a phyloseq object to a metacoder Taxmap via metacoder::parse_phyloseq.
  By default prefers rarefied then batch-removed phyloseq when available. Use when
  the user mentions phyloseq2metacoder, parse_phyloseq, metacoder, Taxmap, or
  heat_tree preparation from phyloseq.
disable-model-invocation: true
---

# Phyloseq → Metacoder

## Purpose

Parse phyloseq to the metacoder object.
By default - use rarefied & with removed batch

Follow: **validation-first**, **reproducibility**, **method-decision-tracking**, **artifact-registry**.

## Input resolution

1. `--rds` if given (must be phyloseq or list with `$phyloseq`)
2. Else if `prefer_rare` (default true): rarefied artifacts
   - `*_rare.rds` / `grazing_phyloseq_rare.rds`
   - or `test/rarefaction-analysis/**/phyloseq_rare_*.rds` (skip `*_plain.rds`)
3. Else if `prefer_batchadj` (default true): batch-removed artifacts
   - `*_batchadj.rds` / `grazing_phyloseq_batchadj.rds`
   - or `test/removebatch/**/phyloseq_batchadj.rds`
4. Else raw count phyloseq (`grazing_phyloseq.rds`)

Unlike alpha/beta, batch-adjusted **relative** tables are allowed (metacoder heat trees use abundances as continuous values).

## Conversion

- `metacoder::parse_phyloseq(ps)` → `Taxmap`
- Optional `--calc-abund true`: attach `data$taxon_counts` via `metacoder::calc_taxon_abund(..., data = "otu_table")`

## Workflow

```
Phyloseq2metacoder:
- [ ] Step 1: Resolve RDS (rare → batchadj → raw by default)
- [ ] Step 2: Validate phyloseq has otu_table + tax_table
- [ ] Step 3: parse_phyloseq → Taxmap
- [ ] Step 4: Optionally calc_taxon_abund
- [ ] Step 5: Save metacoder.rds + phyloseq2metacoder-report.json
```

## Executable

```bash
Rscript .cursor/skills/phyloseq2metacoder/scripts/phyloseq2metacoder.R \
  --outdir test/phyloseq2metacoder/grazing

Rscript .cursor/skills/phyloseq2metacoder/scripts/phyloseq2metacoder.R --self-test
```

Optional: `--rds PATH` `--prefer-rare false` `--prefer-batchadj false` `--calc-abund true`

Thin hook: `.cursor/hooks/phyloseq2metacoder.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `metacoder.rds` | `Taxmap` from `parse_phyloseq` |
| `phyloseq2metacoder-report.json` | Input path, preferences, n_taxa, data slots |
