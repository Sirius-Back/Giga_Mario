# Adapt — examples

## After `@split` (typical)

```bash
# Folds already at data_splits/full/{train,val,test}/<GCF>/
conda run -n caduceus_env python .cursor/skills/adapt/scripts/adapt.py \
  --input data_splits/full \
  --out adapt \
  --window-size 8192
```

Preserves fold labels in `samples.tsv` and writes:

```
adapt/caduceus_ready/train/sequences/*.txt
adapt/caduceus_ready/train/labels.tsv
adapt/caduceus_ready/val/...
adapt/caduceus_ready/test/...
```

## Before `@split` (unsplit panel)

```bash
conda run -n caduceus_env python .cursor/skills/adapt/scripts/adapt.py \
  --input data/reformat/random_full \
  --out adapt \
  --window-size 8192
```

Emits a single unsplit Caduceus-ready tree; run `@split` afterward on genomes **or** on `adapt/` samples only if a future split strategy documents sample-level folds — baseline still expects genome-level `@split` first when folds are required.

## Auto mode

```bash
python .cursor/skills/adapt/scripts/adapt.py --input auto --out adapt --window-size 8192
```

Prefers `data_splits/full` if present and non-empty; else searches configured raw/reformat roots.
