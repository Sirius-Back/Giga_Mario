# Conversion: `raw/` or `prokaryotes/` → `data_ready/` / `ready_small/`

**Producer:** `src/preprocessing.py` (`@adapt`)  
**Date:** 2026-07-27

## Input layouts

### Eukaryotic panel — `raw/`

```
raw/
  fna/     # genomic FASTA (.fna or .fna.gz), one file per GCF
  gtf/     # matching GTF (.gtf or .gtf.gz)
  tpm/     # wide TPM CSVs (header = gene symbols; one data row)
  random_borzoi_expr_file_mappings.csv   # id → genome (GCF) pairing
```

Current eukaryotic panel (2026-07-29): **11** RefSeq assemblies under `fna/` + `gtf/` with matching TPM under `tpm/` (horse = EquCab3.0 `GCF_002863925.1` + derived `SRX19584896.csv`; goat = ARS1.2 `GCF_001704415.2` + derived `SRX6696967.csv`). TB-T2T horse archived (no GEO for that assembly).

Pairing key: `GCF_########.##` prefix shared by FNA/GTF filenames and the mapping `genome` column.

**Run result (`data_ready/`, ±10 kb):** 9 genomes → **199 908** gene windows + **189 143** non-coding = **389 051** samples (~35 GB including `ready.fna` + `caduceus_ready/`).

**Run result (`ready_v2/`, ±2 500 bp, 2026-07-27):** 9 genomes → **199 908** gene windows + **198 384** non-coding = **398 292** samples (~27 GB). Same gene count as `data_ready/`; more non-coding matches with shorter flanks. Skipped `GCF_041296265.1` (missing TPM). Log: `logs/adapt_raw_ready_v2.log`.

### Prokaryotic panel — `prokaryotes/`

```
prokaryotes/
  fna/  gtf/  tpm/
  random_borzoi_expr_file_mappings.csv   # ids = {assembly}_merged
  expr_file_mappings.csv                 # per-sample GEO provenance
```

For Caduceus-prep, use **mean-merged** TPM only:

```bash
conda run -n caduceus_env python src/preprocessing.py \
  --raw prokaryotes --out ready_small \
  --flank 10000 --seed 42 --tpm-merged-only
```

`--tpm-merged-only` restricts pairing to `tpm/{assembly}_merged.csv` (ignores per-sample `GSE*.csv`).

**Run result (`ready_small/`, 2026-07-27):** **12** genomes → **36 445** gene windows + **23** non-coding = **36 468** samples (~583 MB). Dense prokaryotic genomes leave little intergenic space after ±10 kb neighbour trim, so non-coding count is low.

See also `prokaryotes/data.md`.

## Algorithm

1. **Discover** complete bundles (FNA + GTF + local TPM). Abort if none; skip incomplete genomes with notes in `statistics.json`.
2. **CDS genes** — aggregate GTF `CDS` features per `(chrom, gene_id)` → CDS span `[min, max]`. Prefer `gene "..."` symbol for TPM join.
3. **Ideal window** — ±**flank** bp around the CDS (clipped to chromosome). Default **10 000**; `ready_v2/` uses **2 500**.
4. **Large genes** (CDS length > **130 000** bp) — strand-aware crop: **flank before start** + **120 kb** of CDS; record in `large_genes.csv`.
5. **Neighbours** — if another CDS intersects the ±flank (or overlaps the body), **trim** the window to that neighbour’s CDS corner; record in `neighbours.csv`.
6. **Extract** forward genomic DNA; compute **Length** and **GC** (ACGT only).
7. **Gene properties** → rows in `non_coding.csv` (`kind=gene`).
8. **Non-coding** — intergenic complement of gene windows; greedy **1:1** placement matching each gene’s **length** and **GC** (±0.03); `TPM=0`; IDs `NC_<genome>_<chrom>_<idx>`; append to `non_coding.csv` (`kind=non_coding`).
9. **Export** `ready.fna`, `ready.csv`, and `caduceus_ready/all/{sequences/*.txt, labels.tsv}`.

No train/val/test assignment (`@split` owns folds). Seed **42**.

## Output: `data_ready/`

| Path | Format |
|------|--------|
| `ready.fna` | `>Genome\|GeneOrID\|Chr\|Position_start\|Position_end` + sequence |
| `ready.csv` | `Genome\|GeneOrID\|Chr\|Position_start\|Position_end\|TPM` |
| `non_coding.csv` | `GeneOrID\|Chr\|Position_start\|Position_end\|Length\|GC\|kind\|Genome` |
| `neighbours.csv` | Neighbour-trim events |
| `large_genes.csv` | CDS >130 kb crops |
| `caduceus_ready/` | Per-sample `.txt` DNA + `labels.tsv` (continuous TPM) |
| `statistics.json` | Length/GC histograms + per-genome counts |
| `metadata.json` / `README.md` | Provenance |

## Command

```bash
conda run -n caduceus_env python src/preprocessing.py \
  --raw raw --out data_ready --flank 10000 --seed 42

# Eukaryotes, ±2.5 kb windows:
conda run -n caduceus_env python src/preprocessing.py \
  --raw raw --out ready_v2 --flank 2500 --seed 42

# Prokaryotes (merged TPM only):
conda run -n caduceus_env python src/preprocessing.py \
  --raw prokaryotes --out ready_small --flank 10000 --seed 42 --tpm-merged-only
```

Smoke: add `--genomes GCF_000001405.40 --max-genes 80`.

SLURM wrapper: `src/sbatch/preprocess_raw.sbatch`.

## Skill entry

`@adapt` delegates this conversion to `src/preprocessing.py` (see `.cursor/skills/adapt/SKILL.md`).
