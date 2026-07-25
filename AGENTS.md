# AGENTS

## Main MDs

| File | Role |
|------|------|
| `todo.md` | Task status (TODO / READY / RUNNING / BLOCKED / FAILED / RECOVERABLE / COMPLETED / SKIPPED) |
| `method-decision.md` | Method choices + rationale |
| `artifact-registry.md` | Register every generated deliverable |
| `monitoring-report.md` | Job monitoring output |

Prefer `docs/artifact-registry.md` when `docs/` exists.

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
