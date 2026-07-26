# Conversion: `raw/` → `data_ready/`

**Producer:** `src/preprocessing.py` (`@adapt`)  
**Date:** 2026-07-27

## Input: `raw/` structure

```
raw/
  fna/     # genomic FASTA (.fna or .fna.gz), one file per GCF
  gtf/     # matching GTF (.gtf or .gtf.gz)
  tpm/     # wide TPM CSVs (header = gene symbols; one data row)
  random_borzoi_expr_file_mappings.csv   # id → genome (GCF) pairing
```

Current panel (2026-07-27): **10** RefSeq assemblies under `fna/` + `gtf/`; **9** usable TPM files under `tpm/` (mapping lists `SRX19584896` for `GCF_041296265.1`, but that CSV is absent → genome skipped).

Pairing key: `GCF_########.##` prefix shared by FNA/GTF filenames and the mapping `genome` column.

**Run result:** 9 genomes → **199 908** gene windows + **189 143** non-coding = **389 051** samples in `data_ready/` (~35 GB including `ready.fna` + `caduceus_ready/`).

## Algorithm

1. **Discover** complete bundles (FNA + GTF + local TPM). Abort if none; skip incomplete genomes with notes in `statistics.json`.
2. **CDS genes** — aggregate GTF `CDS` features per `(chrom, gene_id)` → CDS span `[min, max]`. Prefer `gene "..."` symbol for TPM join.
3. **Ideal window** — ±**10 000** bp around the CDS (clipped to chromosome).
4. **Large genes** (CDS length > **130 000** bp) — strand-aware crop: **10 kb before start** + **120 kb** of CDS; record in `large_genes.csv`.
5. **Neighbours** — if another CDS intersects the ±10 kb flanks (or overlaps the body), **trim** the window to that neighbour’s CDS corner; record in `neighbours.csv`.
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
```

Smoke: add `--genomes GCF_000001405.40 --max-genes 80`.

SLURM wrapper: `scripts/preprocess_raw.sbatch`.

## Skill entry

`@adapt` delegates this conversion to `src/preprocessing.py` (see `.cursor/skills/adapt/SKILL.md`).
