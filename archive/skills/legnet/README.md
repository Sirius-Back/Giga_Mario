# LegNet — README

**human_legnet** training entry (Caduceus-parallel skill): `legnet_ready/*.tsv` →
`runs/legnet/<name>/` with Lightning train + Caduceus-like `logs/` for `@train-viz`.

Prep data with `@legnet-adapt` first (`src/legnet_preprocess.py`).

## Quick start

```bash
conda run -n legnet python -m src.legnet \
  --data-path legnet_ready/GCF_000001405.40_folds1to10.tsv \
  --out runs/legnet/demo_GRCh38 \
  --epochs 20 --device 0 --demo \
  --use-shift --reverse-augment
```

## Outputs

| Path | Role |
|------|------|
| `logs/train_metrics.jsonl` | Epoch curves for `@train-viz` |
| `best_model/` | Best `val_pearson` checkpoint |
| `final_model/` | Last-epoch checkpoint |
| `metrics_summary.md` | Test pearson/spearman/mse/… |
| `model_2_1/` | Upstream Lightning dump (demo) |

## Docs

- `.cursor/skills/legnet/SKILL.md`
- `wiki/legnet_conversion.md`
- Vendor: `software/human_legnet`
