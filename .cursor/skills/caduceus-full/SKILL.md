---
name: caduceus-full
description: >-
  End-to-end Caduceus pipeline without adapt: write/reuse src/runs/caduceus_full.py
  which imports src.splits + src.caduceus + src.train_viz (split → train → zs → viz).
  Use for /caduceus-full.
disable-model-invocation: true
---

# Caduceus Full

## Required input

* **ready** data already prepared (`ready/` / `data_ready/`) — **no `@adapt`**
* **raw/** available for split strategies that need genomic metadata
* split strategy in `splits/*.md` (default `random`)

If `ready/` is missing → **stop** (missing-data-policy). Do not run `@adapt` inside this skill.

## What it does

1. **Write / update** `src/runs/caduceus_full.py` (thin imports only)
2. **Exec** that script (re-runnable later **without subagents**)
3. Pipeline inside the script:
   * `/split` → `src.splits.random.run_random_split` → `<out-root>/splits/{M1,M2,zero_shot}`
   * `/caduceus` → `src.caduceus` on **M1** (TPM) then **M2** (predict M1 fold)
   * zero-shot eval of M1 on holdout genomes (default: human)
   * `/train-viz` → `src.train_viz.viz.main` on M1 (and M2 + compare)

## Code-first contract (LOCKED)

```
@caduceus-full:
  1. Confirm raw/ + ready/ present (already adapted)
  2. WRITE/UPDATE src/runs/caduceus_full.py (imports only — no reimplemented trainers)
  3. EXEC: python -m src.runs.caduceus_full --strategy random --out-root output/random \
         --epochs-m1 10 --epochs-m2 5 --zs-genomes human --seed 42
  4. REUSE the same script for future runs (no subagents required)
```

Do **not** reimplement split / caduceus / train-viz logic here. Import their `run` / `main` APIs.
Do **not** run the full `@split` skill/do-fast graph — only `run_random_split` on ready data.

## Exact command

```bash
conda run -n caduceus_env python -m src.runs.caduceus_full \
  --strategy random \
  --raw raw \
  --ready ready \
  --out-root output/random \
  --seed 42 \
  --epochs-m1 10 \
  --epochs-m2 5 \
  --zs-genomes human \
  --nproc 4

# Smoke
python -m src.runs.caduceus_full --max-samples 32 --epochs-m1 1 --epochs-m2 1 --no-m2 --nproc 1

# Multi-GPU: --nproc N uses torch.distributed.run -m src.caduceus (same trainer code)
```

| Flag | Default | Notes |
|------|---------|-------|
| `--strategy` | `random` | `splits/<id>.md` |
| `--out-root` | `output/random` | All artifacts under this tree |
| `--raw` / `--ready` | `raw` / auto | Ready must already exist |
| `--epochs-m1` / `--epochs-m2` | **10** / **5** | Per model |
| `--zs-genomes` | `human` | Alias → `GCF_000001405.40`; holdout from M1/M2 |
| `--nproc` | all CUDA devices | `1` = in-process `run()` |
| `--seed` | **42** | |
| `--skip-split` / `--skip-train` / `--skip-viz` / `--skip-zs` | off | Resume helpers |
| `--no-m2` | off | TPM-only path |
| `--max-samples` | none | Smoke cap |

## Resolved paths

| Param | Path |
|-------|------|
| Split | `<out-root>/splits/{M1,M2,zero_shot}/` |
| Train M1 | `<out-root>/runs/M1/` |
| Train M2 | `<out-root>/runs/M2/` |
| ZS eval | `<out-root>/zs_eval/` |
| Viz | `<out-root>/figures/{M1,M2,compare}/` |
| Report | `<out-root>/report.md` |

## Stage map

```mermaid
flowchart TD
  in[raw/ + ready/ already adapted] --> write[Write src/runs/caduceus_full.py]
  write --> exec[Exec python -m src.runs.caduceus_full]
  exec --> split["src.splits: M1 TPM + M2 + ZS holdout"]
  split --> t1["src.caduceus: train M1"]
  split --> t2["src.caduceus: train M2"]
  t1 --> zs["M1 eval on zero_shot"]
  t1 --> viz["src.train_viz: M1 / M2 / compare"]
  t2 --> viz
  zs --> report["out-root/report.md"]
  viz --> report
```

## Workflow checklist

```
caduceus-full:
- [ ] Confirm ready/ (no adapt) + raw/
- [ ] Update src/runs/caduceus_full.py if wiring must change
- [ ] Exec python -m src.runs.caduceus_full …
- [ ] Verify <out-root>/{splits,runs,figures,zs_eval,report.md}
- [ ] method-decision + artifact-registry
```

## Rules

- **No `@adapt`** — windows/TPM already in `ready/`
- Never invent data, metrics, or fold labels
- Orchestrator = **small imports + `run`/`main` calls** only
- Defaults: **M1 10 ep / M2 5 ep**, seed **42**, ZS = human genome
- TPM train follows `metrics.md` (via `src.caduceus`); viz via `src.train_viz`
- Future re-runs: exec `src/runs/caduceus_full.py` directly — **no subagents required**

## Coordination

| Module | Role |
|--------|------|
| `src/runs/caduceus_full.py` | This pipeline |
| `src/splits/` | `/split` code API only |
| `src/caduceus.py` | `/caduceus` |
| `src/train_viz/` | `/train-viz` |
| `src/preprocessing.py` | Prior `@adapt` only (not invoked here) |

## Additional resources

- [examples.md](examples.md)
- [workflow.md](workflow.md)
- Script: [`src/runs/caduceus_full.py`](../../src/runs/caduceus_full.py)
