---
name: prepare-prompt
description: >-
  Create or update executable task files under ./todo/*.md from a user request
  or from entries in the main todo.md; link those files into todo.md; validate
  via verify-todo. Does not author the large project architecture — that is
  @generate-todo. Planning only — never executes. Use when turning a prompt
  into ./todo/ specs or fleshing out todo.md rows into task files.
disable-model-invocation: true
---

# Prepare Prompt

## Purpose

Create **executable task specifications** under `./todo/*.md` and link them to the main `todo.md`.

This skill owns **task files**, not the high-level project architecture.

| Owner | Artifact |
|-------|----------|
| `@generate-todo` | Main living **`todo.md`** (phases, milestones, hierarchical plan) |
| **`@prepare-prompt`** | **`./todo/*.md`** executable specs + link rows into `todo.md` |
| `@verify-todo` | Dependency graph linking `todo.md` ↔ `./todo/*.md` |

If `todo.md` is missing or the user needs a large architecture / roadmap first, invoke or require **`@generate-todo`** before creating task files. Do not invent a full hierarchical `todo.md` here.

This skill never executes project tasks.

Follow project rules: **validation-first**, **missing-data-policy**, **method-decision-tracking**, **scientific-integrity**, **task-status**, **artifact-registry**.

Discover skills/rules dynamically from `.cursor/skills/` and `.cursor/rules/` (same discovery principle as `@prompt-orchestrator`).

## Inputs

Inspect:

- user prompt
- `todo.md` (required for linking; obtain via `@generate-todo` if absent)
- every task under `./todo/`
- `dependency-graph.md`
- `dependency-graph.mmd`
- `method-decision.md`
- `docs/artifact-registry.md`
- available project rules
- available project skills

## Workflow

```
Prepare-prompt:
- [ ] Phase 0: Ensure todo.md exists (else @generate-todo)
- [ ] Phase 1: Understand the request
- [ ] Phase 2: Search existing tasks / todo.md rows
- [ ] Phase 3: Dependency analysis
- [ ] Phase 4: Generate task specifications (./todo/*.md only)
- [ ] Phase 5: Link ./todo/*.md into todo.md + registry
- [ ] Phase 6: Invoke @verify-todo — must pass
- [ ] Output: prepare-prompt-report.md
```

---

## Phase 0 — Ensure main todo.md

- If `todo.md` is absent or the request is a **large architecture / full replan** → run or require `@generate-todo` first.
- If `todo.md` exists → proceed; create/update only the `./todo/*.md` files needed for the request.

---

## Phase 1 — Understand the request

Determine:

- objective
- deliverables
- expected outputs
- acceptance criteria
- required inputs
- computational complexity
- execution environment

---

## Phase 2 — Search existing tasks

Determine whether the request:

- already exists;
- partially overlaps;
- extends an existing task;
- depends on previous tasks;
- should be split into multiple tasks.

Never create duplicated tasks.

Prefer **update** or **extend** over new IDs when overlap is clear. Split only when deliverables or skills differ enough to need separate acceptance criteria.

---

## Phase 3 — Dependency analysis

Construct explicit dependencies using:

- `todo.md`
- dependency graph (`dependency-graph.md` / `.mmd` from `@verify-todo` — do not rebuild the graph here)
- existing task specifications

Infer:

- prerequisites
- downstream tasks
- reusable outputs (via artifact-registry.md when present)

If the graph is missing or stale, note that Phase 6 `@verify-todo` will regenerate it — still declare Dependencies in new task files from todo analysis.

---

## Phase 4 — Generate task specifications

Create or update task files under:

`./todo/`

using the project's canonical XML-like task schema.

Populate every field.

Determine:

- Status
- Priority
- Dependencies
- Dependents
- Skills
- Rules
- AcceptanceCriteria
- Execution metadata

Schema and examples: [task-schema.md](task-schema.md).

### Status

Use **only**: `TODO` | `READY` | `RUNNING` | `BLOCKED` | `FAILED` | `RECOVERABLE` | `COMPLETED` | `SKIPPED`.

New tasks: usually `TODO` or `BLOCKED` (if deps incomplete) or `READY` (if all prerequisites COMPLETED and inputs available).

### Skills and rules

Select the **minimum necessary** skills from the discovered registry (metadata match). Apply all relevant project rules. Do not hardcode a closed skill list.

### File naming

`todo/T-<id>-<short-slug>.md` (e.g. `todo/T-2.1-dataset-audit.md`). Preserve existing IDs when updating.

---

## Phase 5 — Link into todo.md

Update `todo.md` **only to link** each new/updated `./todo/*.md` (stable IDs, paths, `Depends on:`). Preserve manual notes, milestones, and hierarchy authored by `@generate-todo`.

Also update:

- dependency fields inside the task files
- artifact registry if required (this skill’s report)

Do **not** rewrite the architectural structure of `todo.md`. Do not modify source code or scientific outputs.

---

## Phase 6 — Validation

Invoke:

`@verify-todo`

The generated tasks must successfully validate before completion.

If `@verify-todo` reports **Blocking** issues: fix task specs if the failure is from this skill’s edits; re-run verify-todo. If still Blocking, fail prepare-prompt and return the dependency report — do not claim success.

---

## Outputs

Generate:

`prepare-prompt-report.md`

Prefer `docs/prepare-prompt-report.md`. Template: [prepare-prompt-report-template.md](prepare-prompt-report-template.md).

Summarize:

- created tasks
- modified tasks
- detected dependencies
- required skills
- required rules
- suggested execution order

---

## Execution Rules

- Never execute tasks.
- Never modify source code.
- Never perform scientific analyses.
- Always reuse existing tasks whenever possible.
- Always invoke verify-todo before completion.

Allowed writes: `./todo/**/*.md`, `todo.md` (link/status rows only — not full architecture rewrite), `prepare-prompt-report.md` / `docs/prepare-prompt-report.md`, `artifact-registry.md` / `docs/artifact-registry.md`.

## Coordination

| Skill | Role |
|-------|------|
| `@generate-todo` | Authors main `todo.md` architecture — run first when missing or for large replans |
| `@edit-prompt` | Modify existing `./todo/*.md` plan in place (freeze, propagate, rollback analysis) |
| `@verify-todo` | Validate graph linking `todo.md` ↔ `./todo/*.md` (required) |
| `@prepare` | Later: execution plan for **existing** todos (no task creation) |
| `@do` | Later: execute READY tasks — never called here |
| `@synchronize-todo` | Optional status sync vs repo — not a substitute for Phase 5 |

## Artifact registration

Instead of creating standalone reports in arbitrary locations, require every generated artifact to be registered inside `artifact-registry.md` (prefer `docs/artifact-registry.md`).

Each registry entry must contain:

- artifact
- producer skill
- generation date
- purpose
- status
- downstream consumers

Every generated report, graph, manifest or checkpoint must be registered immediately after it is written.

Update existing rows when regenerating the same path; mark replaced paths `superseded`.

Format: [artifact-registry-template.md](../_shared/artifact-registry-template.md). Project rule: `artifact-registry` (alwaysApply).

## Additional resources

- Task file schema: [task-schema.md](task-schema.md)
- Report template: [prepare-prompt-report-template.md](prepare-prompt-report-template.md)
