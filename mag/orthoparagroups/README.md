# Orthoparagroups (mammals11)

FASTAs extracted from `ready_legnet/MARKED` for Ensembl Compara homology components with `n_nodes > 10` and `paralog_edges > 5`.

## Inputs

| Path | Role |
|------|------|
| `ready_legnet/ID.csv` | gene symbol + GCF → numeric MARKED ID |
| `raw/gtf/*_genomic.gtf` | NCBI GeneID ↔ symbol |
| `mag/homology_graph/maps/gene2ensembl.gz` | NCBI GeneID ↔ Ensembl |
| `mag/homology_graph/edges.tsv.gz` | ortholog/paralog graph |

## Rebuild

```bash
python mag/src/orthoparagroups/prepare_maps.py \
  --id-csv ready_legnet/ID.csv --gtf-dir raw/gtf \
  --outdir mag/homology_graph/maps

mag/src/orthoparagroups/orthoparagroups \
  --edges mag/homology_graph/maps/edges_extract.tsv \
  --nodes mag/homology_graph/maps/nodes_extract.tsv \
  --marked-dir ready_legnet/MARKED \
  --outdir mag/orthoparagroups \
  --min-nodes 11 --min-paralog-edges 6 --seed 42
```

## Outputs

- `cluster_*.fna` — headers tagged `ortholog` or `paralog`
- `clusters.tsv` — cluster property table

## Figures

```bash
python -m src.run.orthoparagroups.plot_clusters \
  --clusters mag/orthoparagroups/clusters.tsv \
  --outdir mag/orthoparagroups/figures
```

cnsplots: `Figure_01`–`Figure_10` (PDF/SVG/PNG). Altair: matching `*_altair` HTML/VL/PNG plus `Figure_11` species median occupancy.
