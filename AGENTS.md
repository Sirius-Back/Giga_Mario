# AGENTS

## Main MDs

| File | Role |
|------|------|
| `todo.md` | Task status (TODO / READY / RUNNING / BLOCKED / FAILED / RECOVERABLE / COMPLETED / SKIPPED) |
| `method-decision.md` | Method choices + rationale |
| `artifact-registry.md` | Register every generated deliverable |
| `monitoring-report.md` | Job monitoring output |
| `docs/adapt.md` | `@adapt` Caduceus-prep overview |
| `docs/caduceus_format.md` | Caduceus fine-tune format notes |
| `metrics.md` | TorchMetrics suite for Caduceus/expression epoch logging |
| `wiki/conversion.md` | raw → data_ready conversion |
| `wiki/split.md` | raw+ready → `src/splits` folds |
| `src/preprocessing.py` | `@adapt` entry (ready panel) |
| `src/splits/main.py` | `@split` dispatcher (`python -m src.splits.main`) |
| `src/caduceus.py` | `@caduceus` fine-tune (`python -m src.caduceus`) |
| `src/metrics_logging.py` | `metrics.md` TorchMetrics helpers |
| `src/train_viz/` | `@train-viz` figures (`python -m src.train_viz`) |
| `src/runs/caduceus_full.py` | `@caduceus-full` orchestrator |
| `src/sbatch/` | SLURM wrappers → `src/` modules |

Prefer `docs/artifact-registry.md` when `docs/` exists.

## Caduceus data pipeline

```
raw / reformat
      │
      ├─(Caduceus-like)──► @adapt ──► @split (regions + TPM) ──► @caduceus
      │
      └─(other)──► @data if missing → convert/regionize
                      │
                      ├─ need Caduceus windows? ──► @adapt ──► @split ──► @caduceus
                      └─ already region-split? ──► align predictions (TPM; no gene→0) ──► @caduceus
```

- **`@split`** assigns **genomic regions** and **linked predictions** (TPM by default) to train/val/test per `splits/*.md`. Does not invent data.
- **`@adapt`** builds CDS±10 kb DNA windows + continuous TPM via `src/preprocessing.py` → `data_ready/` / `ready/`. Runs **before** `@split` when input is Caduceus-like.
- **`@caduceus`** trains/evaluates on region folds + labels; epoch logs must follow **`metrics.md`**.
- **Linkage:** every region has a prediction; region with no gene → prediction **0**.
- **`@caduceus-full`** writes/executes `src/runs/caduceus_full.py`: **no adapt** (uses `ready/`); `/split` (M1 TPM + M2) → `/caduceus` ×2 → `/train-viz`. Re-runnable without subagents. See `.cursor/skills/caduceus-full/SKILL.md`.

## `splits/{}.md`

One file per split strategy (e.g. `splits/random.md`). Shared shape:

| Section | Content |
|---------|---------|
| Frontmatter | `id`, `name`, `aliases` |
| `# Description` | What the strategy does |
| `# Split` | Roles for `train` / `validation` / `test` / `zero_shot` (and extras if any) |
| `# Implementations` | Per-model blocks: `name`, `url`, `paper`, `split_location`, `run`, `notes` |
| `# References` | Sources |

More `splits/*.md` files will be added; keep this structure.

**Training advances:** If the target model supports extras (cross-validation, leave-one-out, etc.), implement them in **both** `@split` and `@caduceus`. Do not document in split MDs only.

## Conditional rules

- **Model training** — read and follow [`.cursor/rules/model-train.mdc`](.cursor/rules/model-train.mdc) when asked to train or fine-tune a model (e.g. Caduceus).
