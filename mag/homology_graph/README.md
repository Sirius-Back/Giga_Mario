# Homology graph (raw mammals-11)

Built from Ensembl Compara release 116 `protein_default` dumps for the 11 species in `raw/`.

## Edge table

`edges.tsv.gz` columns:

| column | meaning |
|--------|---------|
| gene1 | Ensembl gene stable ID |
| genome1 | Ensembl production name |
| gene2 | Ensembl gene stable ID |
| genome2 | Ensembl production name |
| relation | `ortholog` or `paralog` |

Undirected; endpoint order is canonicalized. Genes without any ortholog/paralog edge are omitted.

## Rebuild

```bash
python -m src.run.homology_graph.build_mammals11 \
  --ensembl-data mag/ensembl/data \
  --outdir mag/homology_graph
```

## Figures

- `graph_network.png` — grid of mid-sized connected components
- `figures/Figure_0{1-5}_*` — cnsplots (PDF/SVG/PNG) + Altair (HTML/VL/PNG)
