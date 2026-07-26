---
name: network
description: >-
  Microbial association networks from phyloseq: coexistence (SparCC),
  NetCoMi (SpiecEasi intended; SparCC for speed), tidygraph chord
  (aRchiteutis df2chord), and NetCoMi-igraph viz. Use when the user mentions
  network, SparCC, NetCoMi, SPRING, SpiecEasi, chord, df2chord, coexistence,
  or microbial graph.
disable-model-invocation: true
---

# Network

## Purpose

Build and plot microbial association networks from a phyloseq object:

| `--method` | Association / viz | Notes |
|------------|-------------------|--------|
| `netcomi` (**default**) | NetCoMi `netConstruct` — intended **SpiecEasi**; **SparCC** kept for speed (`--measure sparcc`) | igraph: node~phylum, edge~signed assoc; labels = **`geom_shadowtext` + ggrepel positions** (or numbered + right legend) |
| `coexistence` | NetCoMi **SparCC** + same igraph viz | SEM coexistence |
| `netcomi-ggraph` / `ggraph` | NetCoMi → tidygraph → circular `ggraph` (`tidygraphize` from Metaanalyse) | `geom_edge_arc0` + taxonomy display labels |
| `tidygraph` / `chord` | Pearson cor chord (tip_rank-aware fonts) | aRchiteutis `df2chord` pattern |
| `igraph` | Alias → NetCoMi SparCC + igraph viz | Combined with NetCoMi |

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **slurm-execution-policy**, **artifact-registry**.

## Input resolution

1. `--rds` if given (prefer rarefied counts)
2. Else rarefied candidates / grazing fixtures
3. **Re-finalize taxonomy** if placeholders / dotted `make.unique` tips (`uncultured.3`, `Family.1`)
4. Keep taxa with **mean relative abundance > 0.01%** (`--min-mean-rel 1e-4`); optional `--top-n` cap

## Defaults

| Parameter | Default |
|-----------|---------|
| Method | `netcomi` |
| Measure | `sparcc` (speed; use `--measure spieceasi` when practical) |
| Abundance filter | mean rel > `1e-4` (0.01%) |
| SparCC / coexistence `thresh` | `0.3` |
| Chord `k_means` | `5` |
| Chord `coenf_level` | `0.7` |
| Label mode | `shadowtext` (`geom_shadowtext` + repel positions; `--label-mode numbered` for numbered nodes + right legend) |
| Seed | `123` |

## Workflow

```
Network:
- [ ] Step 1: Resolve phyloseq; finalize taxonomy; abundance filter
- [ ] Step 2: Run selected method(s)
- [ ] Step 3: Write adjacency + figures (igraph/ggraph/chord)
- [ ] Step 4: network-report.json
```

`--method all` runs coexistence + netcomi + netcomi-ggraph + tidygraph.

NetCoMi `*_analyze` / native plots use **taxonomy display labels** (not QIIME ASV hashes).

## Executable

```bash
Rscript .cursor/skills/network/scripts/network.R \
  --outdir test/network/grazing --method netcomi --measure sparcc

Rscript .cursor/skills/network/scripts/network.R --method tidygraph
Rscript .cursor/skills/network/scripts/network.R --method coexistence --label-mode numbered
Rscript .cursor/skills/network/scripts/network.R --self-test

sbatch .cursor/skills/network/scripts/network.sbatch
```

Thin hook: `.cursor/hooks/network.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `adjacency_*.tsv` | Association / adjacency matrix |
| `network_*.pdf` / `.png` | Method-specific figures |
| `network-report.json` | Method, params, paths |

## Vendored code

`df2chord` is copied from aRchiteutis — see [vendor/aRchiteutis/SOURCE.txt](vendor/aRchiteutis/SOURCE.txt). Chord plotting uses tip_rank-aware fontface (species/genus italic; family+ plain).

## Additional resources

- SEM / NetCoMi / igraph notes: [reference.md](reference.md)
