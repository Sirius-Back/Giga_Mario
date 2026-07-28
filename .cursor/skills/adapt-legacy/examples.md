# Adapt — examples

## Full raw panel

```bash
conda run -n caduceus_env python src/preprocessing.py \
  --raw raw --out data_ready --flank 10000 --seed 42
```

## Smoke (one genome, capped genes)

```bash
conda run -n caduceus_env python src/preprocessing.py \
  --raw raw --out data_ready_smoke \
  --genomes GCF_000001405.40 --max-genes 80
```

See `wiki/conversion.md` for I/O contracts.
