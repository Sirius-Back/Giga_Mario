---
name: edit-prompt
description: >-
  Modify an existing project plan without executing tasks: impact analysis,
  freeze completed stages, edit ./todo/*.md specs, propagate downstream changes,
  rollback analysis (RECOVERABLE only), and validate via verify-todo. Use when
  the user changes scope, objectives, or dependencies of existing todos.
disable-model-invocation: true
---

# Edit Prompt

## Purpose

Modify an existing project plan without executing project tasks.
This skill edits one or more existing task specifications under ./todo/ according to a new user request while preserving completed work whenever possible.
The skill performs impact analysis, freezes completed project stages, propagates required downstream changes, validates the dependency graph and prepares the updated project for execution.
This skill never performs scientific analyses or executes project tasks.

Follow project rules: **validation-first**, **missing-data-policy**, **method-decision-tracking**, **scientific-integrity**, **task-status**, **artifact-registry**.

Discover skills/rules dynamically from `.cursor/skills/` and `.cursor/rules/`.

| Owner | Artifact |
|-------|----------|
| `@generate-todo` | Main living **`todo.md`** (architecture — edit only when modification requires it) |
| **`@edit-prompt`** | **Modify** existing **`./todo/*.md`** + propagate links in `todo.md` |
| `@prepare-prompt` | **Create** new `./todo/*.md` from scratch (not modify-in-place) |
| `@verify-todo` | Rebuild/validate dependency graph after edits |

## Workflow

```
Edit-prompt:
- [ ] Phase 1: Determine modification scope
- [ ] Phase 2: Freeze stable stages
- [ ] Phase 3: Edit tasks
- [ ] Phase 4: Dependency propagation
- [ ] Phase 5: Rollback analysis
- [ ] Phase 6: Validate (@verify-todo)
- [ ] Phase 7: Outputs (edit-prompt-report.md)
```

==================================================
Inputs
==================================================

Inspect:

- user prompt
- todo.md
- every task under ./todo/
- dependency-graph.md
- dependency-graph.mmd
- method-decision.md
- docs/artifact-registry.md
- project reports
- available project rules
- available project skills

==================================================
Phase 1 — Determine Modification Scope
==================================================

Identify:

- which existing task(s) are affected;
- whether tasks should be edited, split, merged or deleted;
- downstream tasks affected by the modification;
- completed stages potentially impacted.

Generate an explicit impact summary.

==================================================
Phase 2 — Freeze Stable Stages
==================================================

Unless explicitly overridden by the user:
Freeze every completed project stage occurring before the earliest affected task.
Frozen stages must not be modified.

If the user specifies a freeze point:
Freeze every stage up to and including that stage.

Record frozen tasks in the report.

Use **only** task statuses from the project vocabulary: `TODO` | `READY` | `RUNNING` | `BLOCKED` | `FAILED` | `RECOVERABLE` | `COMPLETED` | `SKIPPED`. Treat **COMPLETED** tasks in frozen stages as immutable unless the user explicitly overrides.

==================================================
Phase 3 — Edit Tasks
==================================================

Modify only the required task specifications.

Allowed operations:

- edit task metadata;
- edit objectives;
- edit acceptance criteria;
- edit execution metadata;
- edit dependencies;
- split tasks;
- merge tasks;
- create additional downstream tasks;
- remove obsolete downstream tasks.

Never edit frozen tasks.

Task file schema: [../prepare-prompt/task-schema.md](../prepare-prompt/task-schema.md).

==================================================
Phase 4 — Dependency Propagation
==================================================

Determine all downstream consequences.

Update:

- dependency graph;
- todo.md;
- dependent task specifications.

Only downstream tasks may be regenerated.
Frozen upstream tasks remain immutable.

Do **not** rebuild `dependency-graph.md` / `dependency-graph.mmd` here — `@verify-todo` is the sole graph owner (Phase 6).

==================================================
Phase 5 — Rollback Analysis
==================================================

Determine whether any completed non-frozen tasks become invalid because of the modification.

If rollback is required:

List every task requiring rollback.
Estimate rollback cost.
Do not automatically rollback.

Instead mark those tasks:

Status = RECOVERABLE

unless explicitly instructed by the user.

==================================================
Phase 6 — Validate
==================================================

Invoke:

verify-todo

Pass explicit context describing:

- modified tasks;
- frozen tasks;
- downstream changes;
- rollback candidates.

verify-todo must rebuild the dependency graph and validate the updated execution plan.

If `@verify-todo` reports **Blocking** issues: fix edits if the failure stems from this skill; re-run verify-todo. If still Blocking, fail edit-prompt and return the dependency report — do not claim success.

==================================================
Phase 7 — Outputs
==================================================

Generate:

edit-prompt-report.md

Prefer `docs/edit-prompt-report.md`. Template: [edit-prompt-report-template.md](edit-prompt-report-template.md).

Include:

- modified tasks;
- frozen tasks;
- downstream tasks updated;
- rollback candidates;
- dependency graph changes;
- execution order changes;
- affected artifacts;
- required next actions.

==================================================
Execution Rules
==================================================

- Never execute project tasks.
- Never modify source code.
- Never modify frozen tasks.
- Preserve completed work whenever possible.
- Prefer downstream propagation over rollback.
- Rollback must never occur automatically.
- Always invoke verify-todo before completion.
- Always regenerate dependency-graph.md and dependency-graph.mmd after successful edits.
- Register all modified planning artifacts in artifact-registry.md.

Allowed writes: `./todo/**/*.md`, `todo.md` (links/status/dependency fields for affected tasks), `edit-prompt-report.md` / `docs/edit-prompt-report.md`, `artifact-registry.md` / `docs/artifact-registry.md`.

## Coordination

| Skill | Role |
|-------|------|
| `@prepare-prompt` | Create **new** task specs from free-form prompts — not in-place plan edits |
| `@generate-todo` | Main `todo.md` architecture when large structural replan is needed |
| `@verify-todo` | Rebuild/validate graph after edits (required Phase 6) |
| `@prepare` | Re-sync execution plan after successful edit-prompt |
| `@do` | Execute updated READY tasks — never called here |
| `@synchronize-todo` | Optional repo↔status sync after edits |

## Artifact registration

Register every generated or materially changed planning artifact in `docs/artifact-registry.md` immediately after write:

- artifact, producer skill, generation date, purpose, status, downstream consumers

Update existing rows when regenerating the same path; mark replaced paths `superseded`.

Format: [artifact-registry-template.md](../_shared/artifact-registry-template.md). Project rule: `artifact-registry` (alwaysApply).

## Additional resources

- Report template: [edit-prompt-report-template.md](edit-prompt-report-template.md)
- Task schema: [../prepare-prompt/task-schema.md](../prepare-prompt/task-schema.md)
