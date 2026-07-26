# Adapt — README

Mandatory **Caduceus-prep** stage: `raw/{fna,gtf,tpm}` → CDS±10 kb DNA windows + matched non-coding + continuous TPM → **`data_ready/`**.

Does **not** create train/val/test folds (`@split` owns that).

## Quick start

```bash
conda run -n caduceus_env python src/preprocessing.py \
  --raw raw --out data_ready --flank 10000 --seed 42
```

Or via skill wrapper:

```bash
conda run -n caduceus_env python .cursor/skills/adapt/scripts/adapt.py \
  --raw raw --out data_ready
```

## Outputs (`data_ready/`)

| File | Role |
|------|------|
| `ready.fna` | Windows; header `Genome\|GeneOrID\|Chr\|start\|end` |
| `ready.csv` | Same + `TPM` |
| `non_coding.csv` | Gene + non-coding Length/GC |
| `neighbours.csv` | Neighbour-trim events |
| `large_genes.csv` | CDS >130 kb crops |
| `caduceus_ready/` | `.txt` sequences + `labels.tsv` for `/caduceus` |
| `statistics.json` | Distributions + per-genome counts |

## Docs

- `wiki/conversion.md` — algorithm + I/O
- `docs/adapt.md`, `docs/caduceus_format.md`
- Legacy ±200 bp path: `scripts/adapt_legacy.py` (`adapt.py --legacy`)
