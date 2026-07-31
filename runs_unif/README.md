# runs_unif/

Unified aligned-suite outputs. Legacy trees under `runs/run{i}` and
`src/runs/run{i}` stay **read-only**.

## Layout

| Kind | Path |
|------|------|
| New code | `src/runs_unif/run{i}_{model}_{split}[_params]/` |
| New artifacts | `runs_unif/{model}/run{i}_{model}_{split}[_params]/` |

## Active: run2_legnet_random

```bash
# CPU: rewrite split 3:1:1 + materialize (no GPU)
conda run -n legnet --no-capture-output \
  python -m src.runs_unif.run2_legnet_random.continue_from_split split_only=true

# Full: after split_done, waits for 4 GPUs (else 2) then direct+adversarial train
conda run -n legnet --no-capture-output \
  python -m src.runs_unif.run2_legnet_random.continue_from_split
```

| Stage | Settings |
|-------|----------|
| Split | from `runs/run2/split.csv`; ≈3:1:1 (train↔val swap) |
| Direct | min 15 / max 30 epochs; early-stop patience 10; mice ZSV |
| Adversarial | random; max 10 epochs; early-stop patience 5 |
