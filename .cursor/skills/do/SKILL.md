---
name: do
description: >-
  Primary execution engine: run prepared TODO waves via subagents, verify-todo,
  prompt-orchestrator, code-review, monitor, and project-auditor with safe
  parallelism and checkpoints. Use when the user asks to execute, run the
  project, or invoke /do after prepare.
disable-model-invocation: true
---

# Do

## Purpose

Execute the prepared scientific project by coordinating specialized subagents.

This skill is the primary execution engine of the project. It delegates all specialized work to existing project skills, coordinates execution, maximizes safe parallelism within resource limits (throughput over agent count), validates intermediate results, maintains project consistency and continues until every reachable task has completed or execution becomes blocked.

This skill must never duplicate functionality implemented by other project skills.

Follow **all active project rules** (including **task-status**, **slurm-execution-policy**). Resume from the latest checkpoint when restarted.

Prerequisites: prefer a successful `@prepare` run; always run `@verify-todo` before each cycle regardless.

## Canonical execution workflow

**Never bypass this execution cycle.**

```
1. project-auditor
2. verify-todo  →  if Blocking: TERMINATE
3. Determine executable tasks
4. For each executable task (resource-capped parallel batch):
      one execution subagent:
        4.1 prompt-orchestrator
              ↓
        4.2 code-review
              if Critical or Major → return to 4.1
              ↓
        4.3 if computational jobs launched → monitor (status only)
              until completion or error (default 2h unless user override)
              if error → @debug (if Safe) → return to 4.1
              ↓
        4.4 project-auditor
              if Critical → return to 4.1
              ↓
        4.5 Mark task COMPLETED
5. Synchronize todo.md, dependency graph, artifact registry, checkpoints
   + concise human-readable progress report
→ Repeat cycle from step 1 until no executable tasks remain
   (or verify-todo Blocking terminates)
```

### 1. Launch project-auditor

Launch a dedicated subagent.

Invoke: `@project-auditor`

Inspect project structure, `todo.md`, `./todo/`, `method-decision.md`, dependency graph, reports, documentation, reproducibility, generated outputs.

Generate an updated audit before the main loop continues.

If audit reports **P0 Critical** blockers that prevent any task from running, stop and report — do not continue until resolved or user overrides.

### 2. Launch verify-todo

Launch a dedicated subagent.

Invoke: `@verify-todo`

Responsible for rebuilding and validating the dependency graph and generating `dependency-graph.md` / `dependency-graph.mmd`.

**If Blocking issues exist: Terminate execution.** Return the dependency report. Do not continue.

Never bypass verify-todo.

### 3. Determine executable tasks

Determine every task whose dependencies have been satisfied.

Identify: executable tasks; blocked tasks; completed tasks; parallelizable tasks.

**Source of truth:** `dependency-graph.md` from `@verify-todo` — do not rebuild the graph here.

A task is executable when Status is **READY** (or becomes READY after verify-todo), all prerequisites are **COMPLETED**, and required inputs exist (or will be produced by this task per plan). Never treat TODO, BLOCKED, FAILED, SKIPPED, or RUNNING as newly executable.

If **no executable tasks remain**, exit the loop → Final Report (success or blocked by incomplete/non-READY work).

### 4. Launch one execution subagent for every executable task

Before launch, estimate CPU / memory / runtime per task; **avoid oversubscription**; prefer **throughput over agent count** (see Resource gating below). Launch only the resource-approved concurrent batch; defer the rest to the next cycle.

Assign **one** executable task to each approved subagent.

Synchronize the batch only after every running task in the batch has finished. Dependent tasks must never begin before prerequisites complete successfully.

Each execution subagent **repeatedly** performs:

#### 4.1 prompt-orchestrator

Invoke: `@prompt-orchestrator`

Execute the assigned TODO task.

Pass: task ID, `./todo/<task>.md` metadata, expected inputs/outputs, skills listed in the subtask.

Set Status **RUNNING** while active.

#### 4.2 code-review

Launch a **NEW** independent subagent.

Invoke: `@code-review`

Compare the implementation against: the original TODO task; the generated prompt; acceptance criteria; `method-decision.md`.

**If Critical or Major findings exist: Return to 4.1.**

Never bypass code-review.

**Retry limit:** at most **3** cycles of 4.1↔4.2 per task unless user directs otherwise; then mark Status **FAILED** (or **RECOVERABLE** if later retry is expected) and continue other tasks.

#### 4.3 monitor (computational jobs only)

**If computational jobs were launched** (SLURM, Nextflow, Snakemake, long-running scripts, Docker, Apptainer or similar):

Launch another independent subagent.

Invoke: `@monitor`

**`@do` is the sole invoker of `@monitor` under project execution.** `@prompt-orchestrator` must not launch `@monitor` (prevents double monitoring).

**Monitor until completion or escalation.**

**Duration inheritance:** when `@do` invokes `@monitor`, the monitoring duration is **inherited from do**. The default inherited duration is **2 hours** unless explicitly overridden by the user when invoking `/do`. Pass this duration to `@monitor` — do not rely on monitor’s direct-invoke default (30 minutes).

`@monitor` only checks job status / progress. It does **not** recover or resubmit.

**If monitor escalates an error:**

1. Invoke **`@debug`** when the failure looks Safe/infrastructure-recoverable.
2. On successful debug + validation → **Return to 4.1**.
3. If debug is Unsafe/Impossible or fails → mark Status **RECOVERABLE** or **FAILED** as appropriate; do not invent success.

Skip 4.3 for short non-computational tasks.

#### 4.4 project-auditor

Launch another **NEW** independent subagent.

Invoke: `@project-auditor`

Verify: outputs; reproducibility; documentation; generated reports; consistency; generated artifacts.

**If Critical findings remain: Return to 4.1.**

Never bypass project-auditor.

Prefer task-scoped / delta audit when possible.

#### 4.5 Mark task COMPLETE

If prompt-orchestrator, code-review, monitor (when applicable), and project-auditor all succeed:

**Mark task COMPLETED** (Status: `COMPLETED`; checkbox `[x]`).

Update `todo.md` and corresponding `./todo/*.md`.

Every completed task must pass every verification stage before downstream execution begins.

On unrecoverable failure after retries: Status **FAILED**. If monitor escalates and `@debug` can repair later: Status may be **RECOVERABLE** until repaired.

### 5. Synchronize and progress report

Synchronize:

- `todo.md`
- dependency graph (invoke `@verify-todo` so graph artifacts stay current)
- artifact registry (`artifact-registry.md`)
- checkpoints (`docs/do-checkpoint.md`, persisted reports)

Generate a **concise human-readable progress report** (`docs/do-progress.md`). Template: [progress-template.md](progress-template.md).

Include: COMPLETED / RUNNING / BLOCKED / FAILED / RECOVERABLE / SKIPPED; phase; graph status; parallel/resource batch status; recovered failures; remaining work; next actions.

Surface a short summary to the user in chat after each cycle.

**Repeat this cycle from step 1** until no executable tasks remain (or step 2 terminates on Blocking).

Never bypass this execution cycle.

---

## Resource gating (applies before step 4 launches)

Before launching parallel subagents, estimate for **each** executable task: **CPU**, **Memory**, **Runtime** (from task metadata, `#SBATCH`, logs, complexity).

Avoid oversubscribing: Σ CPU / Σ memory must fit available resources with headroom; respect SLURM QOS/user limits (**slurm-execution-policy**).

Prefer maximizing **throughput** rather than maximizing the number of simultaneously running agents — fewer resource-fit agents; defer excess READY tasks to the next cycle.

If estimates are Unknown, reduce concurrency rather than oversubscribe.

---

## Task statuses

Use **only**: `TODO` | `READY` | `RUNNING` | `BLOCKED` | `FAILED` | `RECOVERABLE` | `COMPLETED` | `SKIPPED`.

Do not introduce additional execution states. See project rule `task-status`.

---

## Checkpoint

After every successful step-5 synchronization:

Persist: `todo.md`, `./todo/*`, `dependency-graph.md`, `dependency-graph.mmd`, `method-decision.md`, `execution-report.md`, generated reports, `docs/do-checkpoint.md`.

Ensure execution can safely resume from the checkpoint after interruption. Never lose project progress. Resume from the latest checkpoint whenever execution is restarted.

---

## Completion

Execution terminates only when:

- every reachable task has completed (no executable tasks remain);

OR

- `verify-todo` reports Blocking issues.

---

## Final Report

Generate the **project-level** report (owned exclusively by this skill):

`docs/execution-report.md`

Do **not** let `@prompt-orchestrator` write this path — per-task reports live at `docs/execution/<task-id>.md` and may be summarized here.

May reuse/extend the per-task report format; template: [execution-report-template.md](execution-report-template.md).

Include:

- completed tasks;
- skipped tasks;
- failed tasks;
- dependency graph summary;
- project audit summary;
- code review summary;
- monitoring summary;
- recovered failures;
- unresolved blockers;
- generated outputs;
- checkpoint information;
- recommendations.

---

## Execution Rules

- This skill is an orchestrator only.
- Never perform specialized work directly.
- Always delegate work to existing project skills.
- **Never bypass the canonical execution cycle** (steps 1→5 with subcycle 4.1→4.5).
- Always maximize safe parallel execution **within estimated CPU/memory/runtime caps**.
- Before parallel launch, estimate resources per task; avoid oversubscription.
- Prefer maximizing throughput over maximizing simultaneous agent count.
- Never bypass verify-todo.
- Never bypass code-review.
- Never bypass project-auditor.
- Use `@monitor` only for computational jobs (sole owner under `/do`); default duration **2 hours** unless user overrides. On monitor error → `@debug` then return to 4.1 when Safe.
- Every completed task must pass every verification stage before downstream execution begins.
- Immediately terminate if verify-todo reports Blocking issues.
- Keep todo.md synchronized at all times; update artifact registry and checkpoints each cycle.
- Preserve checkpoints after every successful execution cycle.
- Resume from the latest checkpoint whenever execution is restarted.

## Subagent assignment guide

| Role | Skill | When |
|------|-------|------|
| Step 1 auditor | `@project-auditor` | Each cycle start + per-task (4.4) |
| Step 2 graph | `@verify-todo` | Each cycle + step 5 sync |
| Task executor | `@prompt-orchestrator` | 4.1 per executable task |
| Reviewer | `@code-review` | 4.2 after each task execution |
| Supervisor | `@monitor` | 4.3 computational jobs only; status-check; default 2h; on error escalate back to this skill |
| Repair | `@debug` | After monitor escalation / RECOVERABLE; before 4.1 retry when Safe |
| Post-task auditor | `@project-auditor` | 4.4 after review/monitor |

One executable task → one execution subagent (Steps 4.1–4.5). Parallelize across independent tasks only, after Step 3 resource gating.

## Coordination

| Skill | Role |
|-------|------|
| `@prepare` | Planning before `@do` |
| `@verify-todo` | Sole dependency graph construction/validation |
| `@prompt-orchestrator` | Per-task execution |
| `@code-review` | Independent correctness check |
| `@monitor` | Job status supervision only (sole invoker under `/do`) |
| `@debug` | Safe repair after monitor escalation or RECOVERABLE |
| `@project-auditor` | Pre-run and post-task audits |
| `@synchronize-todo` | Optional helper when marking complete |

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

- Progress report: [progress-template.md](progress-template.md)
- Final / checkpoint report: [execution-report-template.md](execution-report-template.md)
