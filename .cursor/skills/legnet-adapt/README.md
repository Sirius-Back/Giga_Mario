# LegNet Adapt — README

Mandatory **human_legnet-prep** stage: `raw/{fna,gtf,tpm}` → TSS-centered 200 bp CRS + lentiMPRA adapter stitch → **`legnet_ready/`**.

Does **not** create project train/val/test folds (`@split` owns Caduceus folds).

## Quick start

```bash
conda run -n legnet python src/legnet_preprocess.py \
  --raw raw --out legnet_ready --crs-bp 200 --seed 42
```

BED mode:

```bash
conda run -n legnet python src/legnet_preprocess.py \
  --bed promoters.bed --fasta genome.fna --out legnet_ready --stitch-adapters
```

## Outputs (`legnet_ready/`)

| File | Role |
|------|------|
| `promoters.bed` | TSS±100 CRS intervals; score=TPM |
| `all.tsv` / `{GCF}.tsv` | human_legnet columns: `seq_id seq mean_value fold rev` |
| `sequences.fa` | Stitched 230 bp FASTA for `fasta_predict.py` |
| `statistics.json` | Counts + skips |
| `metadata.json` | Provenance |

## Docs

- `wiki/legnet_conversion.md` — algorithm + I/O
- Canonical code: `src/legnet_preprocess.py`
- Vendors: `software/human_legnet`, `software/LegNet`
