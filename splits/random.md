---
id: random
name: Random split
aliases:
  - random_split
  - random
---

# Description

Samples (genomic intervals, sequences or examples) are randomly assigned to the train, validation and test sets without considering chromosome, gene, synteny, species or sequence similarity. This is the default split strategy used in many machine learning pipelines and downstream genomic benchmarks.

# Split

train:
- Random subset of all samples.

validation:
- Random subset sampled from the remaining training data (if not provided by the dataset).

test:
- Independent random subset.

zero_shot:
- None.

# Implementations

- name: Caduceus
  url: https://github.com/kuleshov-group/caduceus
  paper: Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling (2024)
  split_location: Downstream dataset loader (`dataset.train_val_split_seed`); pretraining uses predefined interval splits from `human-sequences.bed`.
  run: |
    python -m train \
      experiment=hg38/genomic_benchmark \
      dataset.train_val_split_seed=1
  notes: |
    GenomicBenchmarks provide only train/test. Validation is created by randomly splitting the training set (90/10) using `dataset.train_val_split_seed`. Experiments are repeated over multiple random seeds.

- name: GenomicBenchmarks (used by Caduceus)
  url: https://github.com/kuleshov-group/caduceus
  paper: Grešová et al., 2023
  split_location: Dataset loader during fine-tuning.
  run: |
    dataset.train_val_split_seed=<seed>
  notes: |
    Random train/validation split is generated from the provided training set.

# References

- Caduceus repository: https://github.com/kuleshov-group/caduceus
- Caduceus paper (Supplementary: pretraining dataset and splits): https://pmc.ncbi.nlm.nih.gov/articles/PMC12189541/