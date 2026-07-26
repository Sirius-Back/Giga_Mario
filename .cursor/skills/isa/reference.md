# ISA — grazing Figure 3 reference

Source: `/mnt/tank/scratch/dsmutin/archive/bioinformatics/2025/grazing_article/Article.Rmd` (Indicator species analysis → `fig3`).

## Prep

1. Attach tip labels via `clearify(Genus, Family, Class)` + `clearify_dup`
2. Rarefy (`rarefy_even_depth`, seed 123)
3. Relative abundances for `multipatt` only; rarefied counts for abundance boxplots + sPLS-DA

## multipatt

```r
indicspecies::multipatt(
  x = t(otu_table(ps_rel)),
  cluster = sample_data(ps_rel)[[target]],
  func = "r.g",
  control = permute::how(nperm = 9999)
)
```

Keep `p.value < 0.05`, arrange by `-stat`.

## Panels

| ID | Object | Construction |
|----|--------|--------------|
| a | `fig3a` | Presence/absence by short level → `eulerr::venn` → `plot(fill=…)` |
| b | `fig3b` | ISA membership matrix (indicator columns) → venn |
| c | `fig3c` | Presence of indicator OTUs only → venn |
| d2 | `fig3d2` | Top-N `log10(p.value)` bars |
| d1 | `fig3d1` | Top-N IndVal (`stat`) bars, fill by short level |
| d3 | `fig3d3` | Relative abundance boxplots + `stat_compare_means` |
| e | `fig3e` | `Rtsne` on OTU table; `ggforce::geom_arc_bar` pies by ISA membership |
| f | `fig3f` | `mixOmics::splsda` scores + ellipses |
| g | `fig3g` | Loadings faceted by component, fill by ISA group |

## Combine

Row 1: a|b|c  
Row 2: d2|d1|d3 with `widths = c(1, 5, 5)`  
Row 3: e|f|g with `widths = c(1, 1, 2)`  
Row 4: legends from d2 and d1  
`heights = c(1, 2.2, 1, 0.2)`
