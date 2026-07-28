---
name: pipeline
description: >-
  End-to-end orchestrator around src/run/<run_id>/pipeline.py (dry|run):
  validate inputs, /split, /train, optional /adversarial + adversarial /train.
  Re-runnable without agent. Use for /pipeline.
disable-model-invocation: true
---

# Pipeline (`/pipeline`)

Wrapper around `src/run/<run_id>/pipeline.py` so the full graph can be re-run **without an agent**. Orchestrates `/split`, `/train`, and optional `/adversarial` by writing/executing thin run scripts that import `./src` modules.

## Variants

| Variant | Behavior |
|---------|----------|
| **`run`** | Execute stages end-to-end; monitor long jobs (`@monitor` / SLURM as needed) |
| **`dry`** | Generate code, code-review it, smoketest if needed; **do not** execute full training |

Select via argparse on the orchestrator: `--mode dry|run` (see stub).

## Obligatory inputs

| Input | Meaning |
|-------|---------|
| **`run_id`** | Directory `src/run/<run_id>/` (orchestrator + child scripts live here) |
| **`mode`** | `dry` \| `run` |
| **`data`** | Data panel id |
| **`split`** | Split strategy (`random` for adversarial path) |
| **`train`** | Model: `caduceus` \| `legnet` \| `human_legnet` |
| **`type`** | `regression` \| `classification` |
| **Panel / SPLIT inputs** | Whatever `/split` needs: e.g. `PARSED`, `PREDICT`, `id_csv` / fold / strat as specified; or legacy `raw/`+`ready/` if using `src.splits` path — **must be explicit** |
| **`outdir` / `out-root`** | Artifact root for this pipeline run |

When adversarial is requested, also obligatory:

| Input | Meaning |
|-------|---------|
| **`adversarial=true`** (or equivalent flag) | Enable adversarial branch |
| **`outdir_new`** / adversarial target | Distinct panel root for adversarial combine |
| **Adversarial train model** | Model for second `/train` (often same as direct; must be specified) |

Optional: **ZSV** / holdout genomes for final-model zero-shot-validation on `/train` stages; **`epochs`**, **`seed`** (default 42).

If required features are unspecified → **stop** and list gaps. Do not guess.

## Code-first contract (LOCKED)

```
/pipeline:
  1. Check inputs / all necessary features specified
  2. WRITE/UPDATE src/run/<run_id>/pipeline.py (and child run scripts as needed)
  3. dry → generate + code-review (+ smoketest); no full train
     run → EXEC: python src/run/<run_id>/pipeline.py --mode run …
  4. Later: exec the same pipeline.py without subagents
```

Prefer calling `src.pipeline.*` modules. Child scripts follow `/train` and `/adversarial` naming:

- `{data}_{split}_{train}_direct.py`
- `{data}_{split}_adversarial.py`
- `{data}_{split}_{train}_adversarial.py`

## Stage order

```
1. validate inputs
2. /split          → split-predict + split (or src.splits when Locked)
3. /train          → direct model on SPLIT
4. /adversarial    → if specified (combine + random split)
5. /train          → adversarial model on adversarial SPLIT (if specified)
```

Visualization + TensorBoard are owned by `/train` (reuse Caduceus TB + `src.train_viz`). ZSV final-model eval when specified on train stages.

## dry vs run

**dry**

- Emit/update `pipeline.py` + child scripts
- Run `@code-review` (or equivalent) on generated code
- Optional structural smoke (`src.pipeline.train --smoke`, tiny subset) — **not** full epochs
- Exit without launching full training

**run**

- Exec `pipeline.py --mode run`
- Monitor long GPU jobs; register artifacts
- Fail early on missing inputs

## Exact command

```bash
# dry (generate / review path — skill may stop before this if only writing)
python src/run/<run_id>/pipeline.py --mode dry --help

# run
conda run -n caduceus_env python src/run/<run_id>/pipeline.py --mode run \
  --data <data> --split random --train caduceus --type regression \
  --out-root output/<run_id> --seed 42
```

Template stub (importable): [`src/run/run_id/pipeline.py`](../../../src/run/run_id/pipeline.py). Copy/adapt into a concrete `src/run/<run_id>/` for each experiment.

## Workflow checklist

```
pipeline:
- [ ] Confirm run_id, mode (dry|run), data, split, train, type, out-root
- [ ] Confirm panel inputs; adversarial flags/paths if requested; ZSV if requested
- [ ] Write/update src/run/<run_id>/pipeline.py (+ child scripts)
- [ ] dry: code-review (+ optional smoke); do NOT full-train
- [ ] run: exec pipeline.py --mode run; verify SPLIT + runs + figures
- [ ] method-decision + artifact-registry
```

## Coordination

| Skill / module | Role |
|----------------|------|
| `/split` | Fold assignment + materialize |
| `/train` | Direct + adversarial fine-tune + viz + TB + optional ZSV |
| `/adversarial` | Adversarial panel + random re-split |
| `src.pipeline.*` | Stage implementations |
| `src/run/<run_id>/pipeline.py` | This orchestrator |

## Rules

- Reuse `./src` only — orchestrator = imports + `run_*` calls
- `dry` never runs full training
- `run` is re-executable without agents
- Register outputs in `docs/artifact-registry.md`
