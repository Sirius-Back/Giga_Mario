---
id: vgae
name: VGAE pangenome graph split
aliases:
  - gcn_vae
  - vgae_split
---

# Description

Classic **VGAE** (Kipf & Welling) on a pangenome contingency / hash graph with
**edge weights**, **GC%**, and **k-mer composition** as encoder inputs.
Train / test / val are assigned at **region** grain with counts **3:1:1**.

**Homology firewall:** ortholog / paralog / orthogroup / paragroup labels are
**never** fed into the GCN or VAE encoder. They enter only the post-assignment
objective ``L_hom = mean(sd_para) − mean(sd_ortho)`` (`sd_random` from
`split-check-othoparagroup`) and offline checkers.

# Split

train:
- Regions whose VGAE role assignment is `train` (~3/5 of nodes).

validation:
- Regions assigned `val` (~1/5).

test:
- Regions assigned `test` (~1/5).

zero_shot:
- Optional ZSV from `fold.csv` (held out before assign when provided via
  upstream fold tooling).

# Feature → module map

| Signal | Module |
|--------|--------|
| Weighted adjacency | GCN message passing |
| GC + k-mer composition | GCN input `X` |
| Latent `z`, KL, edge recon | VAE head |
| OG/PG | `L_hom` / checker only |

# Implementations

- name: GigaMario classic VGAE split
  url: local toolkit
  paper: Kipf & Welling, VGAE (2016)
  split_location: `src/splits/vgae/` + `src/pipeline/split_predict.py` (`type=vgae`)
  run: |
    # Stage 1 — region graph from runs_unif + ready_legnet wrap
    python -m src.splits.vgae --stage 1 \
      --out VGAE/stage1_region_k5 \
      --graph-dir runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph \
      --marked-dir ready_legnet/MARKED --k 5

    # Stage 2 — exported hash-node graph
    python -m src.splits.vgae --stage 2 \
      --out VGAE/stage2_hash_k5 \
      --graph-dir runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph \
      --marked-dir ready_legnet/MARKED --k 5

    # Pipeline hook (after pack/train path validated):
    # run_split_predict(..., type="vgae", vgae_graph_dir=..., marked_fasta=...)
  notes: |
    Early stop: min_epochs=25, patience=10 on hard L_hom.
    Outputs under VGAE/ with tensorboard/ + train-viz figures/.
    Stage 1 uses capped region–region edges from contingency_graph; Stage 2
    exports the true hash-node graph via pangenome_export_hash_graph.

# References

- Kipf & Welling, Variational Graph Auto-Encoders
- `splits/pangenome.md`, `wiki/architecture.md`
- `split-check-othoparagroup` `sd_random`
