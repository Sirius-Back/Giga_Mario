---
id: vae
name: MLP-VAE k-mer split (no GCN)
aliases:
  - mlp_vae
  - vae_kmer
  - vae_split
---

# Description

Assign train / validation / test with an **MLP-VAE** on region **k-mer composition**
features (no graph, no GCN). Homology (OG/PG) never enters the encoder; the
primary train objective is **homology_first** `L_hom` (EMA-normalized recon/KL,
KL anneal, Gumbel-Softmax size-weighted `sd_random`). Classic unnormalized VAE
loss is logged each epoch for comparison but does not drive gradients.

Default roles: **3:1:1** (train:test:val). Early stop on hard `L_hom`
(`min_epochs=25`, `patience=10`).

# Split

train:
- Regions whose MLP-VAE role is `train`.

validation:
- Regions whose MLP-VAE role is `val`.

test:
- Regions whose MLP-VAE role is `test`.

zero_shot:
- Not used by default.

# Implementations

## mlp_vae_kmer_k4 (baseline)

- name: mlp_vae_kmer_k4_lossfix
- url: local
- paper: n/a (baseline vs VGAE)
- split_location: `VAE/mlp_vae_kmer_k4_lossfix/split.csv`
- run: `python -m src.splits.vae --k 4 --features runs_unif/legnet/run11_legnet_kmer_k4/feature_table.csv --out VAE/mlp_vae_kmer_k4_lossfix`
- notes: Features = run11 relative 4-mers (256-d). Pack under `VAE/.../pack/`. TB + train-viz under the same outdir.

# References

- Project VGAE homology_first loss (`src/splits/vgae/`)
- run11 k-mer SBS split (`runs_unif/legnet/run11_legnet_kmer_k4/`)
