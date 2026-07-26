---
name: generate-todo
description: >-
  Generate or update the main hierarchical todo.md from architecture docs,
  specs, and requirements (phases, milestones, dependencies, completion
  criteria). Does not create ./todo/*.md — that is @prepare-prompt. Use when
  the user asks for a project plan, roadmap, large architecture breakdown, or
  living top-level todo list.
disable-model-invocation: true
---

# Generate Todo

Generate the **main project `todo.md`** from a project description, architecture document, specifications, workflow diagrams and user requirements. Decompose large architecture into hierarchical phases/tasks with dependencies, milestones, priorities, estimated complexity, expected outputs and completion criteria.

| Owner | Artifact |
|-------|----------|
| **`@generate-todo`** | Main living **`todo.md`** (architecture / roadmap) |
| `@prepare-prompt` | Executable **`./todo/*.md`** specs linked into `todo.md` |
| `@verify-todo` | Dependency graph across both |

**Do not create or edit `./todo/*.md`.** After `todo.md` exists, use `@prepare-prompt` to materialize executable task files, then `@verify-todo` to validate the graph.

Follow project rules: **validation-first**, **missing-data-policy**, **method-decision-tracking**, **scientific-integrity**, **task-status**.

## Inputs

Read when available:

| Source | Extract |
|--------|---------|
| User description | Goals, scope, deadlines, constraints |
| README / architecture docs | Components, data flow, tech stack |
| Specifications | Functional requirements, acceptance criteria |
| Workflow diagrams | Step order, inputs/outputs per stage |
| Existing `todo.md` | Current status — **update, do not recreate** |
| `method-decision.md` | Locked decisions affecting task scope |
| Repo structure | What already exists vs planned |

If inputs conflict, note conflicts in Planning Gaps — do not silently merge.

## Workflow

```
Todo generation:
- [ ] Step 1: Gather project inputs
- [ ] Step 2: Identify phases and milestones
- [ ] Step 3: Decompose into hierarchical tasks
- [ ] Step 4: Add dependencies, priorities, and complexity
- [ ] Step 5: Define outputs and completion criteria
- [ ] Step 6: Write or update todo.md
- [ ] Step 7: Report planning gaps
```

### Step 1: Gather project inputs

Search project docs, specs, diagrams, and existing todos as above.

### Step 2: Identify phases and milestones

Typical scientific/computational phases (adapt to project):

1. Setup & environment
2. Data acquisition & audit
3. Preprocessing / QC
4. Core analysis
5. Statistics & benchmarking
6. Figures & manuscript
7. Reproducibility & release

Define **milestones** as verifiable checkpoints (e.g., "QC passed for all samples", "benchmark complete").

### Step 3: Decompose into hierarchical tasks

Use 3 levels max for readability:

```
## Phase (Milestone target)
### Task group
- [ ] Task item
```

Each task is **atomic** — one person/session can complete it with a clear done state.

Map workflow diagram steps to tasks 1:1 where possible.

### Step 4: Add dependencies, priorities, and complexity

Per task, specify in metadata (see [todo-template.md](todo-template.md)):

| Field | Values |
|-------|--------|
| **ID** | Stable identifier (e.g., `T-3.2`) |
| **Depends on** | Task IDs that must finish first |
| **Priority** | P0 (blocking) / P1 (core) / P2 (nice-to-have) |
| **Complexity** | S / M / L / XL (relative effort) |
| **Status** | `TODO` / `READY` / `RUNNING` / `BLOCKED` / `FAILED` / `RECOVERABLE` / `COMPLETED` / `SKIPPED` |

Do **not** introduce additional execution states.

Mark **BLOCKED** tasks with reason and unblocking requirement. Use **READY** when dependencies are satisfied but work has not started. Use **SKIPPED** for out-of-scope or obsolete tasks (do not invent other labels).

### Step 5: Define outputs and completion criteria

Every task needs:

- **Expected outputs** — files, reports, figures, commits (paths when known)
- **Completion criteria** — objective checks (not "done when looks good")

Example criterion: "`docs/dataset-audit.md` status Ready; all FASTQ paths validated."

Link tasks to skills when relevant (discover from `.cursor/skills/` metadata).

### Step 6: Write or update todo.md

**If `todo.md` exists:** merge new tasks, preserve COMPLETED items and IDs, update status — **never wipe history**.

**If new:** create at project root `todo.md` using [todo-template.md](todo-template.md).

Include:

- Project summary (2–3 sentences)
- Milestone table with target dates (if provided)
- Hierarchical task list with metadata
- Dependency notes or simple DAG summary for critical path

Format for **living tracking**:

- Checkbox per leaf task (`[x]` only when Status is `COMPLETED`)
- Status line updatable in place
- Last-updated date at top

### Step 7: Report planning gaps

When requirements are underspecified, append **Planning Gaps** section or separate `todo-gaps.md`:

- Missing decision, spec, or resource
- Which tasks are BLOCKED
- What user must provide

Do not invent deadlines, resource estimates as facts, or scope not in inputs — mark as **TBD** or **Assumed (confirm)**.

## Deliverables

Default output:

1. **`todo.md`** — living top-level project task list (architecture)
2. **Planning Gaps** — underspecified items (embedded or separate)
3. **Critical path** (optional) — shortest dependency chain to primary milestone

**Out of scope:** `./todo/*.md` task specification files — hand off to `@prepare-prompt`.

## Maintenance mode

When user says "update todo" or marks tasks complete **at the architecture level**:

- Change status and checkbox only for affected tasks in `todo.md`
- Add new tasks with new IDs; never reuse IDs
- Move COMPLETED tasks to a **Done** subsection optionally, but keep IDs for traceability
- For new executable specs under `./todo/`, invoke `@prepare-prompt` (do not write those files here)

## Coordination

| Skill | Role |
|-------|------|
| `@prepare-prompt` | Creates `./todo/*.md` and links them into `todo.md` |
| `@edit-prompt` | Modifies existing `./todo/*.md` in place (not main architecture rewrite) |
| `@verify-todo` | Builds/validates graph after task files exist |
| `@synchronize-todo` | Repo↔status sync for existing `todo.md` |
| `@prepare` / `@do` | Downstream execution prep / run — not called here |

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

- File format and examples: [todo-template.md](todo-template.md)
