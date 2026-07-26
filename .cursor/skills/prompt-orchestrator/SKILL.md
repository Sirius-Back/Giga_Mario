---
name: prompt-orchestrator
description: >-
  Single-task executor: run exactly one assigned TODO task by discovering project
  skills/rules, delegating within that task, selecting SLURM/local environments,
  recovering failures, and writing docs/execution/<task-id>.md. Not a project
  scheduler — use @do for waves, graph order, other TODO tasks, and
  docs/execution-report.md.
disable-model-invocation: true
---

# Prompt Orchestrator

## Purpose

Orchestrate **exactly one assigned task**.

Interpret the assigned task (from `@do` or the user), determine the work required **for that task only**, invoke the appropriate project skills, coordinate within-task execution, recover from failures when possible, and produce a task-scoped execution report.

This skill is **not** a project-level scheduler. Project scheduling, TODO-graph inspection, global execution order, and launching other TODO tasks belong **exclusively** to `@do`.

Never perform specialized work directly when an existing skill is available. Delegate to the most appropriate skills and execution environments **within the assigned task**.

Follow **all active project rules** (discover under `.cursor/rules/` — see [skills-map.md](skills-map.md#project-rules)).

## Scope boundaries

### This skill does

- Execute the **currently assigned** task only
- Discover skills/rules and plan **within-task** steps
- Delegate to subordinate skills needed for that task
- Select environments, submit jobs, recover, validate, and report for that task

### This skill must never

| Forbidden | Owner |
|-----------|--------|
| Schedule project execution (waves, batches, multi-task loops) | `@do` |
| Inspect the **complete** TODO graph to decide what runs next | `@do` / `@verify-todo` |
| Determine **project** execution order across TODO tasks | `@do` |
| Launch **other** TODO tasks (siblings, dependents, or READY queue items) | `@do` |

`prompt-orchestrator` orchestrates **only** the currently assigned task.

If invoked without a single assigned task ID / `./todo/<task>.md`, require that assignment (or a single explicit user task) before proceeding — do not expand into project-wide TODO scheduling.

## Orchestration checklist

```
Assigned-task execution only:
- [ ] Phase 0: Discover installed project skills + project rules
- [ ] Phase 1: Understand the assigned task
- [ ] Phase 1b: Classify governance (rules / skills / both)
- [ ] Phase 2: Plan within-task execution (apply rules; min skills)
- [ ] Phase 3: Delegate work (skills only if needed)
- [ ] Phase 4: Select execution environment
- [ ] Phase 5: Observe this task’s jobs/steps (lightweight; do **not** invoke @monitor)
- [ ] Phase 6: Automatic recovery via @debug if needed
- [ ] Phase 7: Validation
- [ ] Phase 8: Final report (task-scoped)
```
---

## Phase 0 — Discover installed project skills and rules

**Required before selecting any subordinate skill.**

Remove every assumption of a fixed skill list. Dynamically discover every installed project skill before execution.

### Skill registry inspection

1. Inspect the project skill registry at `.cursor/skills/`.
2. Discover every available skill: each subdirectory containing `SKILL.md`.
3. For each skill, read frontmatter and infer responsibility from metadata:
   - `name`
   - `description`
   - body headings / stated purpose (only as needed to discriminate)
4. Build an in-session **discovered skill registry** (name → path → responsibility summary).

### Rule registry inspection

1. Inspect `.cursor/rules/*.mdc`.
2. For each rule, read frontmatter (`description`, `alwaysApply`, `globs`) and body enough to know constraints.
3. Build an in-session **discovered rule registry**.
4. Treat `alwaysApply: true` rules as always in scope unless the user explicitly overrides (rare; record if so).

Do **not** use any hardcoded catalog in this skill or in [skills-map.md](skills-map.md) as an exhaustive inventory. Documentation examples are **illustrative only** and must **never** limit skill discovery.

If the skill registry is empty when skills are required, stop and report — do not invent skills. Missing rules: proceed with discovered rules only; do not invent rule text.

---

## Phase 1 — Understand the assigned task

Analyze the **assigned task** (task ID, `./todo/<task>.md`, and any handoff from `@do`).

Determine for **this task only**:

- task objective
- requested outputs
- required inputs
- expected deliverables
- computational complexity
- within-task dependencies (steps/skills), not the project TODO graph
- required software
- required datasets
- potential risks

If the assignment is ambiguous, resolve using the task file and project context needed for that task (README, `method-decision.md`, listed inputs/outputs). Do **not** scan the full TODO graph to pick additional tasks.

Only ask the user for clarification when execution cannot safely continue (**missing-data-policy**).
---

## Phase 1b — Classify governance (rules vs skills)

**Before invoking subordinate skills**, determine whether the requested task is governed primarily by:

- **project rules**;
- **project skills**;
- **both**.

| Governance | When | Action |
|------------|------|--------|
| **Rules only** | Constraints, standards, or policies suffice (e.g. how to write, validate, allocate SLURM, never fabricate) and no specialized multi-step workflow skill is needed | Apply all relevant rules automatically; **invoke no skills** |
| **Skills only** | Request needs a specialized workflow (acquire data, draft Methods, monitor jobs) and rules are ambient constraints | Apply all relevant rules automatically; invoke the **minimum necessary** skills |
| **Both** | Specialized skill(s) required **and** rules materially shape the work | Apply all relevant rules automatically; invoke the **minimum necessary** skills under those constraints |

### Apply rules automatically

- From the discovered rule registry, select every rule whose scope matches the task (always-apply, glob match, or description match).
- Apply them for the entire run — do not wait for a skill to “remember” them.
- Relevant rules bind even when governance is “skills only”.

### Invoke only the minimum necessary skills

- Select skills only if governance is **skills** or **both**.
- Prefer zero skills when rules alone satisfy the request.
- Prefer one skill over many when one discovered skill already orchestrates the rest.
- Never invoke a skill solely because it appears in documentation examples.

Record in the plan and Phase 8 report: `governance: rules | skills | both`, list of applied rules, list of selected skills (possibly empty).

---

## Phase 2 — Plan within-task execution

Construct an execution plan for the **assigned task only** before acting.

Identify:

- **applied rules** (from Phase 1b — all relevant)
- required skills **only if** governance is skills or both — selected from the Phase 0 discovered registry
- **within-task** step order (not project TODO order)
- within-task step dependencies
- opportunities for parallel **sub-steps** of this task
- expected outputs
- validation strategy

Do **not** plan or reorder other TODO tasks. Project order is owned by `@do`.
### Skill selection rules

When selecting subordinate skills (governance ≠ rules-only):

- inspect the project skill registry (Phase 0);
- discover every available skill;
- infer each skill's responsibility from its metadata;
- select the **minimum necessary subset** of skills;
- never assume that built-in examples are exhaustive;
- never skip a better-matching discovered skill because it is absent from documentation examples;
- never duplicate functionality already implemented by a discovered skill.

Whenever possible, minimize duplicated work by reusing existing outputs (manifests, audits, drafts, partial results).

Document the plan — including governance classification, applied rules, **discovered registry snapshot**, and **selected subset** — for Phase 8.

Discovery procedure details: [skills-map.md](skills-map.md).

---

## Phase 3 — Delegate work

If governance is **rules only**, skip skill invocation; continue with rule-constrained orchestration (environment, validation, report) as needed.

Otherwise invoke the selected project skills by reading and following each skill's `SKILL.md`.

**Never duplicate functionality already implemented by another skill.**

**Invoke only the minimum necessary skills.**

Pass each skill only the inputs it needs. Collect deliverables and paths for Phase 7–8.

Record in the execution report: governance mode, every discovered skill name, which were selected (and why), which rules were applied, and why non-selected skills were not used.

---

## Phase 4 — Select execution environment

Automatically determine where each **within-task** step should execute.

Possible environments:

- local execution
- terminal
- Conda environments
- Docker
- Apptainer/Singularity
- SLURM
- existing workflow managers (Nextflow, Snakemake, etc.)

**Large computational jobs** should automatically use SLURM per **slurm-execution-policy** unless the user explicitly requests local execution.

Whenever SLURM is selected:

- estimate CPUs (16 or 32 default; even; up to 64 if justified)
- estimate memory
- estimate wall time
- generate sbatch scripts
- submit jobs
- record job IDs

Prefer Snakemake/Nextflow when the project already defines the step.

If a discovered skill specializes in job supervision, **do not invoke it from this skill** under `/do` — `@do` step 4.3 owns `@monitor` exclusively (prevents double monitoring).

---

## Phase 5 — Observe execution (no `@monitor`)

Observe every job/step launched **for the assigned task** with **lightweight** checks only:

- process started successfully
- required files created (spot-check)
- logs indicate submission succeeded
- this task’s immediate steps completed or handed off to the scheduler

For SLURM: confirm job was accepted (`sbatch` JobID); do **not** run a long supervision loop.

**Never invoke `@monitor` from `@prompt-orchestrator`.** Continuous job supervision belongs exclusively to `@do` (step 4.3) after this skill returns.

For subagents / short local steps: await completion before dependent within-task phases.
---

## Phase 6 — Automatic recovery

If execution fails **within this task** (before returning to `@do`):

Identify the failure.

Possible causes include:

- missing files
- software not installed
- dependency conflicts
- environment activation failure
- SLURM submission failure
- scheduler limits
- insufficient memory
- insufficient CPUs
- timeout
- incorrect parameters
- corrupted inputs

For Safe infrastructure/metadata recovery, **invoke `@debug`** (read its `SKILL.md` and follow it). Do not call `@monitor`.

Additional reasonable recovery examples (only when Safe and not delegated to `@debug`):

- activate correct environment
- regenerate commands
- repair paths
- restart failed steps of **this** task

Retry only when recovery has a reasonable probability of success.

**Never enter infinite retry loops** (max 2 retries for this task unless user directs otherwise).

Never bypass controlled-access or **missing-data-policy** gaps.

Never “recover” by launching a different TODO task.

Clearly label **recovered** vs **unrecoverable** failures in Phase 8.
---

## Phase 7 — Validation

Verify that execution completed successfully (**validation-first**).

Confirm:

- expected outputs exist
- outputs are non-empty
- generated reports are valid
- workflow completed without critical errors
- this task’s consumers (if any listed on the task) would receive expected inputs

If validation fails, report the failure; attempt recovery (Phase 6) or stop.

When **this task’s** progress must be reflected, update its `./todo/<task>.md` status fields as appropriate; do **not** walk the full TODO list to start the next READY task — that is `@do`. Optional sync of this task’s row in `todo.md` via a discovered sync skill is allowed.

When execution strategy involved methodological choices, update `method-decision.md` (**method-decision-tracking**), preferably via a discovered methods-tracking skill if one exists.
---

## Phase 8 — Final report

Generate a **task-scoped** report only:

`docs/execution/<task-id>.md`

Use the assigned task ID as `<task-id>` (e.g. `T-1.2` → `docs/execution/T-1.2.md`). Create `docs/execution/` if missing.

Template: [execution-report-template.md](execution-report-template.md).

**Do not generate or overwrite `docs/execution-report.md`.** That project-level report is owned **exclusively** by `@do`.

Include:

- **assigned task** (ID + summary)
- **governance classification** (rules / skills / both)
- **applied rules** (all relevant, applied automatically)
- within-task execution plan
- **discovered skill registry** (complete list at run time)
- **selected skills** (minimum necessary; may be empty) and selection rationale
- execution environments
- software used
- jobs submitted
- SLURM job IDs
- completed within-task steps
- recovered failures
- remaining issues
- generated outputs
- recommendations

Do **not** report or schedule other TODO tasks as part of this skill’s responsibility.

**Register** `docs/execution/<task-id>.md` in `artifact-registry.md` (prefer `docs/artifact-registry.md`) immediately after writing.

---

## Parallel orchestration (within the assigned task only)

Whenever independent **sub-steps of the assigned task** exist:

- assign them to separate subagents (Task tool);
- execute them concurrently;
- synchronize results before downstream steps of **this** task;
- avoid duplicated work between subagents;
- merge into one task-scoped execution report.

Maximize parallel execution of within-task steps when dependencies allow.

Decide parallel vs sequential order from skill metadata, declared inputs/outputs, and the assigned task — not from a fixed pairing table. Illustrative patterns in [skills-map.md](skills-map.md) are non-binding.

**Never** use parallelism to launch other TODO tasks. Multi-task parallel waves are `@do` only.

---

## Execution rules

- Orchestrate **exactly one** assigned task; never act as a project-level scheduler.
- Never schedule project execution, inspect the complete TODO graph for run order, determine project execution order, or launch other TODO tasks — those belong exclusively to `@do`.
- Always discover the skill and rule registries before delegation.
- Before invoking skills, classify governance as **rules**, **skills**, or **both**.
- Apply all relevant project rules automatically.
- Invoke only the minimum necessary skills (zero when rules-only).
- Always reuse existing project skills when metadata matches the need and governance requires skills.
- Never treat documentation examples as an exhaustive skill list.
- Never duplicate specialized functionality.
- Prefer automation over manual execution.
- Prefer reproducible execution.
- Prefer workflow managers when appropriate.
- Follow all active project rules (discovered under `.cursor/rules/`).
- Stop only when automatic recovery is no longer possible.
- Clearly distinguish recovered failures from unrecoverable failures.
- Record every important execution decision, including governance, applied rules, and skill discovery results.
- Update method-decision.md whenever execution strategy requires methodological choices.
- Update only **this task’s** progress in todo files when needed; leave wave scheduling to `@do`.
- Write only `docs/execution/<task-id>.md` for this run; **never** generate or overwrite `docs/execution-report.md`.
- Register every generated report in `artifact-registry.md`.

## Coordination with `@do`

| Concern | Owner |
|---------|--------|
| Project waves, READY queue, resource-capped batches | `@do` |
| Dependency graph / Blocking terminate | `@verify-todo` (via `@do`) |
| Execute one assigned task (steps 4.1) | `@prompt-orchestrator` |
| Task execution report `docs/execution/<task-id>.md` | `@prompt-orchestrator` |
| Project-level `docs/execution-report.md` | `@do` only |
| Post-task code-review / monitor / auditor | `@do` cycle (4.2–4.4) |

When invoked from `@do`, accept the assigned task and return control after Phase 8 — do not start the next TODO.
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

- Discovery procedure and illustrative examples: [skills-map.md](skills-map.md)
- Final report template: [execution-report-template.md](execution-report-template.md)
