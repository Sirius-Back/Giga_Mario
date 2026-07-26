# Caduceus-full — examples

## Standard (ready panel, random split)

```bash
python -m src.runs.caduceus_full \
  --strategy random \
  --raw raw \
  --ready ready \
  --seed 42 \
  --epochs 10
```

Stages: split (M1/M2) → caduceus M1+M2 → train-viz → `docs/caduceus-full-report.md`.

## Smoke

```bash
python -m src.runs.caduceus_full --max-samples 32 --epochs 1 --no-m2
```

## Resume train only (splits already on disk)

```bash
python -m src.runs.caduceus_full --skip-split --epochs 10
```
