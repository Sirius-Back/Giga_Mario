---
name: taxonomy-tree
description: >-
  Rebuild taxonomy trees from taxon names or NCBI taxids via rentrez → ape →
  ggtree. Use when reconstructing a missing phyloseq tree, building trees from
  taxons/taxids, or when the user mentions taxonomy-tree.
disable-model-invocation: true
---

# Taxonomy Tree

## Purpose

Rebuild a taxonomy tree from taxon names and/or NCBI taxids (rentrez lineage → ape), with codebase sanitize rules (fill NA with last classified rank; as.character → factor before `ape::as.phylo`). Emit lineage tables, Newick, and ggtree figures.

## Workflow

```
Taxonomy tree:
- [ ] Step 1: Ensure mock misc fixtures if defaults missing (@mock-data)
- [ ] Step 2: Resolve names → taxids (or load taxids)
- [ ] Step 3: Fetch lineages; fill NA ranks; sanitize for ape
- [ ] Step 4: Build tree; write nwk/rds + ggtree PNG/PDF + HTML report
```

## Executable

```bash
Rscript .cursor/skills/taxonomy-tree/scripts/taxonomy_tree.R --self-test
Rscript .cursor/skills/taxonomy-tree/scripts/taxonomy_tree.R \
  --taxons test/misc/taxons.json --taxids test/misc/taxids.tsv \
  --outdir test/taxonomy-tree --mode both
```

Thin hook wrapper: `.cursor/hooks/taxonomy-tree.sh`
