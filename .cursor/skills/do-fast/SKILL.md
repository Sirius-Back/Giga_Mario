---
name: do-fast
description: >-
  Fast execution engine: one orchestrator subagent runs verify-todo, task waves,
  monitor/debug, with full @project-auditor only at start and end, brief
  @task-gate between tasks, and @code-review once at the end. Use for /do-fast
  or fast project execution after prepare.
disable-model-invocation: true
---

# Do Fast

## Purpose

Execute the prepared scientific project until every reachable task completes or work is blocked — via **one** orchestrator subagent (not turn-by-turn parent control like `@do`).

**Lean verification profile (default — why “fast”):**

| Gate | When |
|------|------|
| **Full `@project-auditor`** | **Once at start** + **once at end** (before Final Report) |
| **Brief `@task-gate`** | After each task (outputs + AC smoke only) |
| **`@code-review`** | **Once at end** (batch over completed / changed tasks) |
| **`@verify-todo`** | Each wave (graph must stay valid) |
| **`@monitor` / `@debug`** | Computational jobs only, as before |

This cuts the failure mode seen on short pipelines (e.g. `/split` smoketests): six full audits + six code-reviews for local I/O.

This skill must never duplicate functionality implemented by other project skills. Follow **task-status**, **slurm-execution-policy**, and other active project rules. Resume from the latest checkpoint when restarted — **only for the same run** (see [Checkpoint / new run](#checkpoint--new-run)).

Prerequisites: prefer a successful `@prepare` run; always run `@verify-todo` before deciding READY work.

## How to invoke (parent)

1. Optionally surface a one-line start note to the user.
2. If the consumer starts a **new** task graph or new primary `OUT` (e.g. `/split` after a prior EXIT A), **reset** `docs/do-fast-checkpoint.md` first (`start_audit: pending`, new `Run:` id). Do not skip START based on another run’s `start_audit=passed`.
3. Launch **exactly one** orchestrator subagent.
4. Pass **only** the [Single orchestration prompt](#single-orchestration-prompt) below (fill placeholders). Do **not** add a second orchestration message or micromanage the cycle.
5. Wait until an [exit case](#completion). Then present Final Report / blockers to the user.

Parent rules:

- **One sent prompt** — the block under [Single orchestration prompt](#single-orchestration-prompt) is the entire handoff.
- Orchestrator delegates to listed skills (including `@task-gate`); do not invent substitute audits.
- When a specialized skill is **instructions-only** (markdown workflow, no separate runtime agent), the executor may follow that `SKILL.md` and implement via project `src/` — still write `docs/execution/<task-id>.md` and apply task-gate. Do not reimplement other skills’ ownership (e.g. do not replace `@verify-todo` graph building).
- Run-until-done: continuous cycling + `@debug` recovery — not a one-wave shortcut.
- Never bypass **start/end** full auditor, **end** code-review, **per-wave** verify-todo, or **per-task** task-gate (enforced inside the prompt).
- Prefer **chunky** READY tasks (consumers like `@split` should not leave six doc-only micro-tasks).

### Optional override

If the user explicitly asks for **strict / parity-with-`@do`** verification, set in overrides:
`VERIFY_PROFILE=strict` → per-task `@code-review` + per-task `@project-auditor` (legacy heavy loop). Default is **lean**.

## Single orchestration prompt

Send this verbatim as the sole user/task message to the orchestrator subagent (replace `{{…}}`):

```
You are the @do-fast project orchestrator. Debug, recover, and keep cycling until an exit case. Do NOT stop after one wave, one task, or the first failure.

VERIFY_PROFILE: {{VERIFY_PROFILE_OR_LEAN}}
  lean (default): full @project-auditor ONLY at START and END; after each task use brief @task-gate (not full auditor); @code-review ONLY once at END (batch).
  strict: per-task @code-review + per-task @project-auditor (legacy).

EXIT CASES (only legal stops):
  A) every reachable task COMPLETED (no executable/READY work left) AND end-gates passed (end code-review + end full auditor); OR
  B) @verify-todo reports Blocking; OR
  C) START (or END) @project-auditor P0 Critical that prevents ANY task from running (report and stop).
Partial FAILED/RECOVERABLE on some tasks is NOT an exit — continue other READY work and later cycles.

Resume from latest checkpoint if present: docs/do-fast-checkpoint.md (else docs/do-checkpoint.md).
  Resume ONLY if checkpoint Run id matches this invocation’s run (same todo graph / OUT). If Run differs or start_audit is from a prior finished EXIT A for another OUT → treat as NEW run: start_audit pending; run START gates.
User overrides: {{USER_OVERRIDES_OR_NONE}}
Monitor duration inheritance: {{MONITOR_DURATION_OR_2H}}

Canonical cycle (never bypass; repeat until an EXIT CASE):

START (once per run, before the main loop — skip if checkpoint says start-audit already passed THIS run):
  S1. FULL @project-auditor. If P0 Critical blocks all work → EXIT C.
  S2. @verify-todo. If Blocking → EXIT B.

MAIN LOOP:
  1. @verify-todo — rebuild/validate graph. If Blocking → EXIT B.
     Do NOT re-run full @project-auditor here (lean).
  2. Determine executable tasks from dependency-graph.md only.
     Executable = Status READY, prerequisites COMPLETED, inputs exist (or this task produces them).
     Never treat TODO|BLOCKED|FAILED|SKIPPED|RUNNING as newly executable.
     If none → go to END GATES (do not EXIT A until end-gates pass).
  3. Resource-gate the READY set (CPU/mem/runtime; throughput over agent count; no oversubscription; even CPUs; SLURM policy).
  4. For each approved task in the batch, launch one execution path that repeatedly:
     4.1 @prompt-orchestrator — assign one task; Status RUNNING; pass task ID, ./todo/<task>.md, I/O, skills.
     4.2 Code-review: SKIP in lean (deferred to END). In strict only: @code-review; Critical|Major → return to 4.1 (max 3) else FAILED|RECOVERABLE.
     4.3 If computational jobs launched → @monitor (status only; duration = inherited above).
         @do-fast is sole @monitor invoker; @prompt-orchestrator must not launch @monitor.
         On error → @debug when Safe/infrastructure-recoverable → on success return to 4.1;
         if Unsafe/Impossible or debug fails → RECOVERABLE|FAILED; continue others.
         Skip 4.3 if non-computational.
     4.4 Brief @task-gate (lean) — outputs exist/non-empty + AC smoke. Fail → return to 4.1 (max 3) else FAILED|RECOVERABLE.
         In strict: FULL task-scoped @project-auditor instead of (or after) task-gate; Critical → return to 4.1.
         Prefer inline @task-gate (same agent); do not spawn a heavy auditor subagent mid-loop under lean.
     4.5 Mark COMPLETED ([x]) in todo.md + ./todo/*.md only if 4.1 + (4.2 if strict) + 4.3 (if any) + 4.4 succeed.
         Treat mid-run COMPLETED as provisional until END code-review passes.
   Sync barrier: wait for entire batch before next wave.
  5. Synchronize todo.md, @verify-todo graph, artifact-registry.md, docs/do-fast-checkpoint.md;
     write docs/do-fast-progress.md (template: progress-template.md). Brief chat-summary after each wave (not after every micro-step).
     MUST Repeat MAIN LOOP from step 1 — do not return to parent until EXIT A/B/C.

END GATES (when step 2 finds no READY work — before EXIT A):
  E1. Batch @code-review over all tasks COMPLETED this run (or since last end-review). One review pass, not per-task mid-loop.
      Critical|Major → set affected tasks RECOVERABLE or FAILED; return to MAIN LOOP (retry limits apply). Do not invent Pass.
  E2. FULL @project-auditor (end bookend). P0 that blocks declaring success → fix via MAIN LOOP or EXIT C if nothing can run.
  E3. Write docs/execution-report.md (template: execution-report-template.md). Register artifacts. Then EXIT A.

Statuses only: TODO|READY|RUNNING|BLOCKED|FAILED|RECOVERABLE|COMPLETED|SKIPPED.
Orchestrator only — never do specialized domain work; always delegate to skills.
Per-task reports: docs/execution/<task-id>.md — do not let @prompt-orchestrator write docs/execution-report.md.
```

## Canonical execution workflow (reference)

```
START (once):  full project-auditor → verify-todo
MAIN LOOP:
  verify-todo → READY set → resource gate
    → prompt-orchestrator → [monitor?] → task-gate → mark COMPLETED
  sync + progress → repeat
END (once):    batch code-review → full project-auditor → execution-report → EXIT A
```

`VERIFY_PROFILE=strict` restores per-task code-review + per-task project-auditor inside the loop (see prompt).

### START — full project-auditor (once)

Invoke: `@project-auditor` (full).

If **P0 Critical** blockers prevent any task from running → **EXIT C**.

Record in checkpoint: `start_audit=passed`.

### MAIN — verify-todo (each wave)

Invoke: `@verify-todo`. Rebuild `dependency-graph.md` / `.mmd`.

**If Blocking → EXIT B.** Never bypass verify-todo.

### MAIN — determine executable tasks

Source of truth: `dependency-graph.md`.

Executable = Status **READY**, prerequisites **COMPLETED**, inputs exist (or produced by this task).

If **none** → **END GATES** (not immediate EXIT A).

### MAIN — execute batch (4.1–4.5)

Resource-gate first (see below). One task → one execution path. Sync barrier after the batch.

#### 4.1 prompt-orchestrator

Invoke: `@prompt-orchestrator`. Set Status **RUNNING**.

#### 4.2 code-review (lean: skip mid-run)

**Lean:** skip. **Strict:** `@code-review`; Critical|Major → return to 4.1 (max **3** cycles) else FAILED|RECOVERABLE.

#### 4.3 monitor (computational only)

`@do-fast` is the sole `@monitor` invoker. Default duration **2h** unless user override. On error → `@debug` when Safe → return to 4.1; else RECOVERABLE|FAILED.

Skip for short non-computational work.

#### 4.4 task-gate (lean default)

Invoke: `@task-gate` (prefer **inline** checklist).

Fail → return to 4.1 (max **3**) else FAILED|RECOVERABLE.

**Strict:** use task-scoped `@project-auditor`; Critical → return to 4.1.

#### 4.5 Mark COMPLETED

Mark COMPLETED only if required mid-gates passed. Mid-run COMPLETED is **provisional** until END code-review.

### MAIN — synchronize (step 5)

Update: `todo.md`, graph via `@verify-todo`, artifact registry, `docs/do-fast-checkpoint.md`, `docs/do-fast-progress.md` ([progress-template.md](progress-template.md)).

Chat: **one short summary per wave**, not per micro-gate.

### END GATES

1. **Batch `@code-review`** — all tasks completed this run (single pass / single report ok: `docs/code-review/end-run.md` plus optional per-task refs). Critical|Major → demote affected tasks, return to MAIN LOOP.
2. **Full `@project-auditor`** — end bookend. Block success on P0 that invalidates deliverables.
3. **`docs/execution-report.md`** — then **EXIT A**.

---

## Resource gating

Before parallel launch, estimate **CPU / Memory / Runtime** per task. Avoid oversubscription; prefer **throughput over agent count**; even CPUs; respect **slurm-execution-policy**. If Unknown, reduce concurrency.

---

## Task statuses

`TODO` | `READY` | `RUNNING` | `BLOCKED` | `FAILED` | `RECOVERABLE` | `COMPLETED` | `SKIPPED` only.

---

## Checkpoint / new run

After each successful wave sync, persist todos, graphs, method-decision, reports, `docs/do-fast-checkpoint.md` (include `Run`, `start_audit`, `verify_profile`, last completed task ids). Resume safely after interruption **of the same run**.

| Field | Required |
|-------|----------|
| `Run` | Stable id for this invocation (e.g. `full-random-split`, `smoketest-small`) |
| `start_audit` | `pending` \| `passed` (plus audit path when passed) |
| `verify_profile` | `lean` \| `strict` |
| `Completed tasks` | Task ids finished this run |

**New run (parent or consumer responsibility before launch):** when `todo.md` / primary outputs change (new OUT, new T- ids), write a fresh checkpoint with `start_audit: pending` and a new `Run:` value. Archive obsolete `todo/T-*.md` so `@verify-todo` does not mix graphs. Prior EXIT A must not skip START for the new run.

---

## Completion

| Exit | Condition |
|------|-----------|
| **A** | No READY work **and** END GATES passed |
| **B** | verify-todo Blocking |
| **C** | Full auditor P0 prevents any work (start or end with nothing runnable) |

Single-task FAILED / RECOVERABLE is **not** project exit.

---

## Final Report

`docs/execution-report.md` owned **only** by this skill ([execution-report-template.md](execution-report-template.md)). Summarize: tasks, graph, **start + end** audits, **end** code-review, monitoring, recoveries, outputs, checkpoint, recommendations.

---

## Execution Rules

- Orchestrator only — delegate specialized work.
- Parent sends **exactly one** orchestration prompt; subagent runs until EXIT A/B/C.
- **Lean by default:** full auditor bookends; `@task-gate` mid-loop; code-review at end only.
- Never skip verify-todo when selecting READY work.
- Never skip end code-review or end full auditor under lean (unless user set an explicit documented escape hatch).
- `@monitor` only for computational jobs; on error → `@debug` when Safe.
- Maximize safe parallel execution within resource caps; prefer throughput over agent count.
- Keep todo + registry + checkpoints synchronized each wave.

## Ideas baked into this profile (minimize waste)

1. **Bookend audits** — expensive full audits twice, not 2×N.
2. **End batch code-review** — one pass catches systemic issues; mid-run CR rarely changes I/O tasks.
3. **`@task-gate` inline** — no extra subagent spawn for “does the file exist?”.
4. **Provisional COMPLETED** — end CR can reopen; unblocks the pipeline without lying about final quality.
5. **Wave-level chat summaries** — less noise than per-gate narration.
6. **`VERIFY_PROFILE=strict`** — escape hatch when the user wants `@do`-parity gates.
7. **Checkpoint `start_audit=passed`** — avoid re-auditing the whole repo on every **resume** wave of the **same** run (not across different OUTs).
8. **Chunky tasks** — `/split` full panel succeeded with T-1→T-2→T-3 under lean; smoketest micro-tasks (T-1.1…T-3.2) caused gate thrash.
9. **Run-scoped checkpoint** — reset when starting a new split/OUT so prior EXIT A cannot skip START.

## Subagent assignment guide

| Role | Skill | When (lean) |
|------|-------|-------------|
| Start / end auditor | `@project-auditor` | START once + END once |
| Graph | `@verify-todo` | Each wave + sync |
| Task executor | `@prompt-orchestrator` | 4.1 per task |
| Mid gate | `@task-gate` | 4.4 per task (prefer inline) |
| End reviewer | `@code-review` | END batch once |
| Supervisor | `@monitor` | 4.3 computational only |
| Repair | `@debug` | After monitor escalation |

## Coordination

| Skill | Role |
|-------|------|
| `@prepare` | Planning before `@do-fast` |
| `@verify-todo` | Sole dependency graph construction/validation |
| `@prompt-orchestrator` | Per-task execution |
| `@task-gate` | Brief mid-run acceptance gate |
| `@code-review` | End-of-run batch review (lean) |
| `@monitor` | Job status only (sole invoker under `/do-fast`) |
| `@debug` | Safe repair after monitor escalation |
| `@project-auditor` | Full audit at start + end (lean) |
| `@synchronize-todo` | Optional helper when marking complete |
| `@do` | Heavier multi-step parent orchestration (not one-shot) |
| `@split` | Typical consumer; expects lean do-fast by default |

## Artifact registration

Register every generated report, graph, manifest, checkpoint in `artifact-registry.md` (prefer `docs/`). Required fields: artifact, producer skill, generation date, purpose, status, downstream consumers. Update in place; mark replaced paths `superseded`.

Format: [artifact-registry-template.md](../_shared/artifact-registry-template.md).

## Additional resources

- Progress: [progress-template.md](progress-template.md)
- Final report: [execution-report-template.md](execution-report-template.md)
