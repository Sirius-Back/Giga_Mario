---
name: split-check-othoparagroup
description: >-
  Check train/test/val leakage vs orthogroups and paragroups by merging
  runs_unif split.csv with the MARKED ortholog/paralog hash table. Builds
  othologs.csv and paralogs.csv (n_train/n_test/n_val/sd_random). Use when
  the user mentions split-check-othoparagroup, ortholog/paralog split audit,
  othologs.csv, or asks to score splits against Compara homology groups.
disable-model-invocation: true
---

# split-check-othoparagroup

## Purpose

Audit a panel `split.csv` against Ensembl Compara **orthogroups** / **paragroups**
using the prebuilt sorted hash table from `mag/src/orthoparagroups/build_hash_table.py`.

## Obligatory inputs

| Input | Required | Default / notes |
|-------|----------|-----------------|
| **split.csv** | yes | `runs_unif/{model}/{run_id_and_params}/split.csv` (`ID\|train_test\|fold`) |
| **hash table** | yes | `mag/homology_graph/maps/gene_ortho_para_hash.tsv` |
| **outdir** | yes (or infer) | `runs_unif/splits/{model}_{run_id_and_params}/` |

Stop (missing-data-policy) if split or hash table is missing/empty. Do **not** invent group IDs.

## Processing (LOCKED)

```
Инпут: split csv (id, ..., train test val) - посмотри куда она реально кладется по runs_unif/*/*/*
Процессинг:
> считаем id_hash
> сортируем по id_hash
> мерджим с id_hash c таблицей собранной на прошлом шаге
> summarized tables: othologs.csv: orthogroup | n_train | n_test | n_val | sd_random; такая же paralogs.csv
sd_random - отклонение от случайного распределения по группе, которое реально есть в split csv (а не всегда 3:1:1)
```

### Step map → code

| Step | Action | Code |
|------|--------|------|
| locate split | Prefer `runs_unif/{model}/{run}/split.csv` (not nested adversarial unless asked) | path discovery |
| id_hash | FNV-1a 32-bit (`stable_hash`, same as hash table) | `mag/src/split_check_othoparagroup/stable_hash.hpp` |
| sort | sort split rows by `id_hash`, then `id` | `sort_split_by_hash` |
| merge | two-pointer merge with hash table on `id_MARKED_hash` + `id_MARKED` | `merge_and_count` |
| summarize | per orthogroup / paragroup counts; skip empty groups; skip `zsv`/other roles in counts | writes CSVs |
| sd_random | `sqrt(Σ_r (O_r − n·p_r)²)` with empirical `p_train,p_test,p_val` from the split (train+test+val only) | `sd_random_deviation` |

## Binary (run directly or via skill)

```bash
# build
make -C mag/src/split_check_othoparagroup

# infer outdir from runs_unif/{model}/{run}/split.csv
./mag/src/split_check_othoparagroup/split_check_othoparagroup \
  --split runs_unif/legnet/run2_legnet_random/split.csv \
  --hash-table mag/homology_graph/maps/gene_ortho_para_hash.tsv

# explicit outdir
./mag/src/split_check_othoparagroup/split_check_othoparagroup \
  --split PATH/split.csv \
  --hash-table mag/homology_graph/maps/gene_ortho_para_hash.tsv \
  --outdir runs_unif/splits/{model}_{run_id_and_params}
```

CLI flags: `--split`, `--hash-table`, `--outdir`, optional `--model`, `--run-id`.

## Outputs

Under `runs_unif/splits/{model}_{run_id_and_params}/`:

| File | Columns |
|------|---------|
| `othologs.csv` | `orthogroup\|n_train\|n_test\|n_val\|sd_random` |
| `paralogs.csv` | `paragroup\|n_train\|n_test\|n_val\|sd_random` |
| `summary.json` | global role counts/fractions + group counts |

## Agent workflow

1. Validate hash table exists (rebuild with `python3 mag/src/orthoparagroups/build_hash_table.py` if missing).
2. Resolve target run dirs (`runs_unif/{model}/…`); confirm each has `split.csv`.
3. `make -C mag/src/split_check_othoparagroup` if binary missing/stale.
4. Run the binary per split (parallel OK if RAM headroom holds — class `light` / small CPU).
5. Confirm `othologs.csv` + `paralogs.csv` under `runs_unif/splits/{model}_{run_id_and_params}/`.
6. Register outputs in `docs/artifact-registry.md`; append method note only if metric definition changes.

## Rebuild hash table (upstream)

```bash
python3 mag/src/orthoparagroups/build_hash_table.py \
  --id-csv ready_legnet/ID.csv \
  --nodes mag/homology_graph/maps/nodes_extract.tsv \
  --edges mag/homology_graph/edges.tsv.gz \
  --out mag/homology_graph/maps/gene_ortho_para_hash.tsv
```

## Additional resources

- Algorithm detail: [reference.md](reference.md)
