# Network — reference

## Sources

| Path | Role |
|------|------|
| `/mnt/tank/scratch/dsmutin/bee/SEM/main.Rmd` | SparCC `netConstruct`, NetCoMi gcoda, SPRING, igraph centrality |
| https://github.com/dsmutin/aRchiteutis `plots/chord.R` | `df2chord` (cor → circular ggraph) |
| honey `network_clr_matrix` / `spiec_adjacency_matrix` | SpiecEasi glasso (optional future) |

## coexistence (SEM)

```r
full_net <- NetCoMi::netConstruct(
  phyloseq_net,
  measure = "sparcc",
  zeroMethod = "pseudo",
  sparsMethod = "threshold",
  thresh = 0.3,
  seed = 123
)
full_props <- NetCoMi::netAnalyze(full_net, clustMethod = "cluster_fast_greedy", hubPar = "eigenvector")
plot(full_props, rmSingles = TRUE, ...)
```

## netcomi (SEM)

```r
netcomi_net <- NetCoMi::netConstruct(
  tse_or_ps,
  filtTax = "numbSamp", filtTaxPar = list(numbSamp = 0.1),
  measure = "gcoda", dissFunc = "signed",
  thresh = 0.6, sparsMethod = "threshold", seed = 13075
)
g <- SpiecEasi::adj2igraph(abs(netcomi_net$adjaMat1))
plot(g, layout = layout_with_fr(g), ...)
```

## tidygraph / df2chord

Vendored verbatim from aRchiteutis. Input: taxa × samples numeric matrix (rows = taxa). Uses `cor(t(df))`, `igraph::graph_from_adjacency_matrix`, `ggraph` circular arcs.

## igraph (SEM)

```r
g <- SpiecEasi::adj2igraph(abs(adjaMat))
# centrality: degree, betweenness, closeness, eigenvector
# LCC via decompose.graph; size vertices by centrality
```
