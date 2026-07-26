---
name: verify-todo
description: >-
  Single source of truth for the project TODO dependency graph. Construct and
  validate the graph from todo.md and ./todo/, write dependency-graph.md/.mmd,
  block or repair issues, and clear execution. Use before prepare, do, or
  multi-agent runs. Other skills must not rebuild this graph independently.
disable-model-invocation: true
---

# Verify Todo

Validate TODO structure and construct the **complete project dependency graph**.

This skill is the **single source of truth** for the task dependency graph. Other skills must never independently rebuild or validate it — they consume `dependency-graph.md` / `dependency-graph.mmd` produced here.

Inspect `todo.md` and every task under `./todo/`. Construct the graph, validate it, classify issues, and either clear execution to continue or stop with a report.

Follow project rules: **validation-first**, **missing-data-policy**, **scientific-integrity**, **task-status**.

## Inputs

- `todo.md`
- every Markdown file under `./todo/*.md`
- optional: repository state for dependency vs project-state conflicts
- optional: `method-decision.md`, prior `docs/prepare-report.md`

## Workflow

```
Verify todo:
- [ ] Construct dependency graph (todo.md + ./todo/*.md)
- [ ] Write dependency-graph.md + dependency-graph.mmd
- [ ] Validate graph
- [ ] Classify issues (Blocking / Recoverable / Informational)
- [ ] Blocking → dependency-report.md + failure + stop (+ propagate)
- [ ] Recoverable only → @debug → re-validate graph → return control to @prepare / @do
- [ ] Success → report ready for execution
```

Always regenerate `dependency-graph.md` and `dependency-graph.mmd` whenever the task graph changes (including after repair).

---

## Dependency Graph Construction

The skill must become the single source of truth for the project dependency graph.

During execution it must inspect:

- `todo.md`
- every task under `./todo/*.md`

Construct the complete dependency graph from all tasks.

The graph must include:

- task identifiers
- prerequisite dependencies
- downstream dependents
- execution order
- parallelizable branches
- blocking tasks
- terminal tasks (no dependents)
- orphan tasks
- missing references

Generate:

- `dependency-graph.md`
- `dependency-graph.mmd` (Mermaid)

The Mermaid graph should accurately represent all dependencies between tasks.

Prefer paths under `docs/` (`docs/dependency-graph.md`, `docs/dependency-graph.mmd`) when `docs/` exists; otherwise project root.

### Extraction sources

| Source | Extract |
|--------|---------|
| `todo.md` | Task IDs, `Depends on:`, status, outputs, subtask links |
| `./todo/*.md` | Execution metadata: Depends on, Inputs, Outputs, Status, Parallel with |

### Graph element definitions

| Element | Definition |
|---------|------------|
| Prerequisites | Tasks listed in `Depends on` |
| Dependents | Tasks that list this task as a prerequisite |
| Execution order | Topological waves (parallel within a wave when no mutual deps) |
| Parallelizable branches | Tasks in the same wave with no edge between them |
| Blocking tasks | Incomplete tasks that other incomplete tasks depend on |
| Terminal tasks | Nodes with no dependents |
| Orphan tasks | No edges and not explicitly marked independent |
| Missing references | Depends-on / link IDs with no matching task |

Templates: [dependency-graph-template.md](dependency-graph-template.md).

---

## Dependency Validation

Validate the constructed graph.

Verify:

- every referenced dependency exists;
- every referenced dependent exists;
- every dependency points to exactly one task;
- no circular dependencies exist;
- no orphan tasks exist (unless explicitly marked independent);
- every required input is produced by an upstream task;
- no task depends on impossible or mutually exclusive outputs;
- execution order can be constructed;
- no unreachable tasks exist;
- dependency graph is connected where expected.

Also retain prior checks when applicable:

- no task depends on a task that can never complete;
- no dependency conflicts with the current project state.

Classify findings as:

- **Blocking**
- **Recoverable**
- **Informational**

---

## Blocking Issues

Blocking issues include, but are not limited to:

- circular dependency;
- missing dependency;
- dependency on a deleted or SKIPPED task;
- impossible execution order;
- dependency requiring unavailable inputs;
- mutually exclusive dependencies;
- unresolved dependency chains;
- invalid graph structure.

---

## Blocking Behavior

If ANY Blocking issue exists:

Generate:

`dependency-report.md`

The report must include:

- every blocking issue;
- affected tasks;
- explanation;
- graph location;
- suggested fixes.

Immediately terminate execution.

Return a failure status.

If verify-todo was invoked by another skill (including `prepare`, `prepare-prompt`, `do`, `prompt-orchestrator` or any orchestration skill), propagate the failure so that the parent skill also terminates immediately.

Do NOT invoke any additional skills.

Never attempt automatic recovery for Blocking issues.

Prefer path: `docs/dependency-report.md` (or project root if `docs/` absent).

Template: [dependency-report-template.md](dependency-report-template.md).

---

## Recoverable Behavior

If only Recoverable issues are found:

1. Invoke **`@debug`** to repair (when Safe):
   - dependency links;
   - task metadata;
   - graph inconsistencies;
   - task organization.
2. After `@debug` returns, **rerun dependency graph construction and validation** (regenerate `dependency-graph.md` and `dependency-graph.mmd`).
3. Continue only if validation succeeds.
4. **Return control to the caller** (`@prepare` or `@do`). Do **not** invoke `@prompt-orchestrator` — PO must not rebuild project-level execution plans.

Always rerun validation after any automatic repair.

**Limits:** at most **one** repair cycle unless the user requests another. If repair fails or Blocking issues remain after re-validation → Blocking behavior (report + failure + stop; no further skill invocation except writing the report).

If `@debug` is not available, document that in `dependency-report.md`, list recoverable issues, return failure, and stop without inventing repairs.

---

## Success

If no Blocking issues remain:

Report that:

- dependency graph successfully constructed;
- dependency graph validated;
- execution order determined;
- project is ready for execution.

Informational issues may be listed in `dependency-report.md` or the success message but must not block continuation.

Ensure `dependency-graph.md` and `dependency-graph.mmd` reflect the validated graph.

---

## Exit status (for parent skills)

| Result | Status | Parent must |
|--------|--------|-------------|
| **Blocking** | failure | Terminate immediately; do not call further skills |
| **Recoverable → repair failed** | failure | Treat as Blocking |
| **Success** (no Blocking) | success | Continue (e.g. `@prepare` / `@do`); consume graph artifacts |

## Execution Rules

- verify-todo is the only skill responsible for constructing and validating the task dependency graph.
- Other skills must never independently rebuild or validate the dependency graph.
- Always regenerate `dependency-graph.md` and `dependency-graph.mmd` whenever the task graph changes.
- Never attempt automatic recovery for Blocking issues.
- Always terminate immediately on Blocking issues.
- Always rerun validation after any automatic repair.

## Coordination

| Skill | Role |
|-------|------|
| `@generate-todo` | Authors main `todo.md` |
| `@prepare-prompt` | Creates/updates `./todo/*.md` then must call `@verify-todo` |
| `@edit-prompt` | Modifies existing `./todo/*.md` then must call `@verify-todo` with edit context |
| `@prepare` | Invoke `@verify-todo`; consume graph — do not rebuild it; **does not create tasks** |
| `@do` | Invoke `@verify-todo` every cycle; consume graph — never rebuild; resumes after Recoverable repair |
| `@debug` | Repair links / metadata when Recoverable (invoked here) |
| `@prompt-orchestrator` | **Not** called from Recoverable path — single-task only; no project plan rebuild |
| `@synchronize-todo` | Status vs repo — complementary, not a graph substitute |

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

- Graph artifacts: [dependency-graph-template.md](dependency-graph-template.md)
- Failure / success report: [dependency-report-template.md](dependency-report-template.md)
