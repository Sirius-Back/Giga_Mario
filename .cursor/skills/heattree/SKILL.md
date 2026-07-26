---
name: heattree
description: >-
  Build metacoder heat trees from phyloseq or Taxmap, merging taxa to Family by
  default (filter_taxa + reassign). Prefers phyloseq2metacoder Taxmap, else rare
  then batch-removed phyloseq. Use when the user mentions heattree, heat_tree,
  metacoder heat tree, or taxonomic heat-tree plots.
disable-model-invocation: true
---

# Heat Tree

## Purpose

Report the heattree.
By defaut: use the merging taxa at the `family` level

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **artifact-registry**.

## Input resolution

1. `--metacoder` Taxmap RDS if given
2. Else existing `test/phyloseq2metacoder/**/metacoder.rds` (prefer `grazing`)
3. Else `--rds` phyloseq / list-with-`$phyloseq` → `parse_phyloseq`
4. Else rarefied → batch-removed → raw phyloseq (same order as `phyloseq2metacoder`)

## Taxon merge (default Family)

Codebase pattern (`heat_tree_cust` / abundance heat trees):

1. `calc_taxon_abund` → `taxon_counts` + column `total` (row mean) + `leaf` (row sum)
2. Optional `filter_taxa(leaf >= min_leaf)`
3. **Merge to rank:** `filter_taxa(taxon_ranks == <Rank>, supertaxa = TRUE)`  
   (default rank **Family**; OTU observations reassigned to retained taxa)
4. `heat_tree(node_label = taxon_names, node_size = n_obs, node_color = total, layout = davidson-harel)`

Override: `--rank Class` / `Order` / `Genus` (matched case-insensitively to Taxmap ranks).

## Workflow

```
Heattree:
- [ ] Step 1: Resolve Taxmap or phyloseq → parse_phyloseq
- [ ] Step 2: calc_taxon_abund; attach total + leaf
- [ ] Step 3: Filter leaf ≥ min_leaf; merge to Family (default)
- [ ] Step 4: heat_tree → PDF + PNG; save filtered Taxmap
- [ ] Step 5: Write heattree-report.json
```

## Executable

```bash
Rscript .cursor/skills/heattree/scripts/heattree.R \
  --outdir test/heattree/grazing

Rscript .cursor/skills/heattree/scripts/heattree.R --self-test
```

Optional: `--metacoder PATH` `--rds PATH` `--rank Family` `--min-leaf 10` `--subset Alphaproteobacteria,Bacilli`

Thin hook: `.cursor/hooks/heattree.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `heattree.pdf` / `heattree.png` | Publication heat tree |
| `heattree_taxmap.rds` | Filtered Taxmap used for the plot |
| `heattree-report.json` | Input, rank, n_taxa, figure paths |
