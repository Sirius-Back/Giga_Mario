---
name: task-gate
description: >-
  Brief post-task acceptance gate: check declared outputs exist and are
  non-empty, smoke-test acceptance criteria, and flag obvious path/registry
  gaps. Not a full project audit or code review. Use between do-fast tasks
  mid-run, or whenever a lightweight done-check is needed without @project-auditor.
disable-model-invocation: true
---

# Task Gate

## Purpose

Fast, **task-scoped** done-check after `@prompt-orchestrator` finishes a single TODO.

**Not** `@project-auditor` (no repo-wide reproducibility / methodology / publication sweep).  
**Not** `@code-review` (no design/quality deep review).

Owned use-case: mid-run gates inside `@do-fast` so full audits and code-review can stay at the **bookends**.

## When to use

| Use | Skip |
|-----|------|
| After each task in `@do-fast` main loop | Instead of per-task `@project-auditor` |
| Smoke-check outputs before marking COMPLETED | Instead of per-task `@code-review` |
| Quick reopen decision (missing files) | Full scientific health check |

## Inputs

- Task id + `./todo/<task>.md` (AcceptanceCriteria, ExpectedOutputs, Deliverables)
- Filesystem / paths claimed by the task
- Optional: `method-decision.md` only if the criterion explicitly references a Locked choice

## Checklist (keep short — target ≤5 minutes wall / minimal tokens)

```
task-gate:
- [ ] 1. Status RUNNING (or just finished orchestrator) for this task id
- [ ] 2. Every ExpectedOutput / Deliverable path exists
- [ ] 3. Files/dirs non-empty where a non-empty artifact is expected
- [ ] 4. AcceptanceCriteria smoke: each bullet pass/fail with one-line evidence
- [ ] 5. If task said register artifacts — matching row present or noted as deferred to wave sync
```

**Do not:** inventory the whole repo, re-read unrelated skills, rewrite Methods, run `@verify-methods`, or open nested auditors.

## Verdicts

| Verdict | Meaning | Orchestrator action |
|---------|---------|---------------------|
| **Pass** | Criteria met; outputs present | May mark task COMPLETED (provisional until end code-review) |
| **Fail** | Missing output or failed AC | Return to task execution (retry); after limit → FAILED / RECOVERABLE |
| **Warn** | Non-blocking gap (e.g. registry row deferred) | Pass with warning logged; continue |

Only **Fail** blocks COMPLETED.

## Output

Prefer a **short** note (not a full audit essay):

- Append 3–10 lines to `docs/do-fast-progress.md` **or**
- Write `docs/task-gates/<task-id>.md` with: verdict, checklist table, one-line next action

Template:

```markdown
# Task gate: T-X.Y
**Verdict:** Pass | Fail | Warn
**Date:** YYYY-MM-DD

| Check | Result | Evidence |
|-------|--------|----------|
| outputs exist | pass/fail | path… |
| non-empty | pass/fail | … |
| AC smoke | pass/fail | … |

**Notes:** …
```

Register `docs/task-gates/` rows in the artifact registry only if the parent run registers docs routinely; do not create sprawling per-check registries mid-wave.

## Execution mode

**Prefer inline** (same agent as the task executor / orchestrator applies this checklist).  
Launch a dedicated subagent **only** if the parent explicitly asks for isolation.

## Coordination

| Skill | Boundary |
|-------|----------|
| `@do-fast` | Invokes this mid-loop (4.4); full `@project-auditor` at start/end only |
| `@project-auditor` | Full audit — bookends only under do-fast lean profile |
| `@code-review` | End-of-run batch under do-fast lean profile |
| `@verify-todo` | Graph validity — separate from this gate |
| `@prompt-orchestrator` | Produces the work this gate checks |

## Rules

- validation-first, missing-data-policy, task-status, scientific-integrity  
- Never invent missing files as present  
- Never upgrade Warn → Pass by ignoring Fail items  
