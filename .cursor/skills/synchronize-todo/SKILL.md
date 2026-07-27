---
name: synchronize-todo
description: >-
  Reconcile todo.md with repository state by detecting completed, partial,
  obsolete, and missing tasks from code, outputs, and docs. Use when the user
  asks to sync, update, or refresh the todo list from actual project progress.
disable-model-invocation: true
---

# Synchronize Todo

Inspect the entire project repository and compare its current state against todo.md. Detect completed, partially completed, obsolete and missing tasks by analyzing code, documentation, workflows, figures and generated outputs. Update todo.md accordingly while preserving manually written notes. Produce a summary of newly completed work and remaining tasks.

Follow project rules: **validation-first**, **scientific-integrity**, **missing-data-policy**, **task-status**.

Pair with `@generate-todo` for initial todo creation; this skill **reconciles** existing `todo.md` with repo reality.

## Workflow

Copy and track progress:

```
Todo sync:
- [ ] Step 1: Read todo.md and preserve manual notes
- [ ] Step 2: Scan repository for evidence
- [ ] Step 3: Evaluate each task against completion criteria
- [ ] Step 4: Detect gaps and obsolete items
- [ ] Step 5: Update todo.md surgically
- [ ] Step 6: Write sync summary
```

### Step 1: Read todo.md and preserve manual notes

Locate `todo.md` at project root. If missing, stop and recommend `@generate-todo` first.

Before editing, extract and **never delete**:

- Sections marked `## Manual notes` or `<!-- manual -->` … `<!-- /manual -->`
- Freeform paragraphs not matching task template (user commentary between phases)
- Inline `Note:` / `User:` lines on task entries
- Custom fields not in the standard schema (leave untouched)

Parse structured fields: task IDs, checkboxes, Status, Outputs, Done when, Depends on.

### Step 2: Scan repository for evidence

Systematically inspect:

| Evidence type | Locations | Signals completion |
|---------------|-----------|------------------|
| Code / scripts | `src/`, `*.py`, `*.R`, `*.sh` | Implemented functions, committed logic |
| Workflows | `Snakefile`, `*.smk`, Nextflow, WDL | Rules/processes present and wired |
| Config | `config/`, `*.yaml`, env files | Parameters defined |
| Documentation | `docs/`, `README`, audit/methods drafts | File exists and non-empty |
| Data outputs | `results/`, `figures/`, `tables/` | Expected artifacts from task Outputs field |
| Logs / SLURM | `logs/`, `*.out` | Successful run evidence (supporting, not sole proof) |
| Tests / QC | audit reports, benchmark metrics | Pass criteria documented |

Build an **evidence map**: `artifact path → task ID(s)`.

Use `Done when` criteria as primary truth — file existence alone is insufficient if criteria require quality checks (e.g., audit status Ready).

### Step 3: Evaluate each task

Assign detection result:

| Result | Criteria |
|--------|----------|
| **COMPLETED** | All Done when conditions verified in repo |
| **READY** | No work started, but dependencies and inputs are satisfied |
| **RUNNING** | Work in progress; some outputs exist or criteria not fully met |
| **TODO** | Not started; dependencies not yet satisfied or no evidence |
| **BLOCKED** | External blocker or unmet dependency still valid |
| **FAILED** | Clear failure; Done when cannot be met without redesign |
| **RECOVERABLE** | Inconsistent or failed state that sync/repair should fix |
| **SKIPPED** | Superseded, out of scope, or intentionally not run |
| **Missing from todo** | Repo work exists with no matching task (propose new entry) |

Update only:

- Checkbox `[ ]` → `[x]` when **COMPLETED**
- `Status:` field
- Optional `Synced:` line with date and evidence path (append; do not remove user notes)

Do **not** mark COMPLETED on inference alone — cite evidence path in sync summary.

For partial progress (**RUNNING**), keep checkbox unchecked; set `Status: RUNNING`; append `Progress:` line if helpful.

### Step 4: Detect gaps and obsolete items

**Missing tasks** — propose new task entries with new IDs (coordinate ID scheme with existing todo). Add under appropriate phase; mark `Added by sync:`.

**Obsolete / out-of-scope tasks** — set `Status: SKIPPED`; append `Obsolete reason:`; do not delete (preserve history). Examples: replaced workflow, descoped analysis, duplicate of another task.

**Milestone rollup** — update milestone Status to **COMPLETED** when all dependent tasks are COMPLETED (otherwise TODO / RUNNING / BLOCKED as appropriate).

Conflicts (todo says COMPLETED but artifacts missing): flag as **Regression**; prefer Status **RECOVERABLE** pending confirmation — do not invent other states.

### Step 5: Update todo.md surgically

Rules:

- Bump **Last updated** date
- Merge edits into existing structure — **never recreate the file**
- Preserve task IDs, manual notes, and user commentary verbatim
- Append sync metadata; don't overwrite `Note:` / `User:` lines
- Update Summary section (2–3 sentences) to reflect current phase

Optional: append `## Sync log` with dated one-line entries (or write `docs/todo-sync-log.md` if user prefers lean todo.md).

### Step 6: Write sync summary

Deliver separate **Sync Summary** using [sync-template.md](sync-template.md):

1. Newly COMPLETED since last sync
2. RUNNING / partial (what remains)
3. New tasks added
4. SKIPPED tasks
5. Remaining critical path (P0 TODO/READY/BLOCKED)
6. Evidence gaps needing user confirmation

## Deliverables

1. **Updated `todo.md`**
2. **Sync Summary** — `docs/todo-sync-summary.md` or inline at bottom under `<!-- sync-summary -->` if user prefers

## Coordination

| Skill | When |
|-------|------|
| `@generate-todo` | No todo.md or full architecture replan needed |
| `@prepare-prompt` | Need `./todo/*.md` specs for existing todo.md rows |
| `@dataset-auditor` | Verify audit-task completion criteria |

Apply project rule **validation-first** when confirming outputs meet Done-when checks (rule, not a skill).

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

- Sync summary template: [sync-template.md](sync-template.md)
- Task format reference: [../generate-todo/todo-template.md](../generate-todo/todo-template.md)
