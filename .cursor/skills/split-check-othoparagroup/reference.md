# split-check-othoparagroup — reference

## Hash table schema

`mag/homology_graph/maps/gene_ortho_para_hash.tsv` (sorted by `id_MARKED_hash`):

`id_MARKED|id_MARKED_hash|id|genome|orthogroup|orthogroup_hash|paragroup|paragroup_hash`

- Empty `id` / groups ⇒ MARKED gene with no Compara map (ignored for group summaries).
- Multi-Ensembl MARKED IDs ⇒ multiple hash rows; each group is counted.

## Group definitions

- **orthogroup**: connected component on ortholog edges only (full `edges.tsv.gz`).
- **paragroup**: connected component on paralog edges only.

## sd_random

From the input split, among rows with `train_test ∈ {train,test,val}`:

```
p_r = n_r / (n_train + n_test + n_val)
n   = n_train_g + n_test_g + n_val_g   # within group
sd_random = sqrt( Σ_r (O_r − n·p_r)² )
```

`zsv` / other labels are excluded from both global fractions and per-group counts.

## Source layout

```
mag/src/split_check_othoparagroup/
  stable_hash.hpp
  types.hpp
  io.hpp / io.cpp
  merge_summarize.hpp / merge_summarize.cpp
  main.cpp
  Makefile
  split_check_othoparagroup   # binary after make
```
