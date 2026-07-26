---
name: monitor
description: >-
  Check status of running computational jobs (SLURM, Snakemake, local processes)
  for a fixed duration; estimate progress when indicators exist; on error hand
  off to @do (project orchestrator). Does not recover, resubmit, or redesign
  workflows. Default 30 minutes when invoked directly; when invoked by @do,
  inherit do’s duration (default 2 hours unless user override). Use when
  monitoring jobs or pipelines — only @do launches this skill under /do.
disable-model-invocation: true
---

# Monitor

## Purpose

Check the status of computational jobs for a given monitoring window.

Poll SLURM jobs, workflow managers, and long-running local processes. Record health and progress. Produce `monitoring-report.md`. On any failure or stall that blocks completion, **stop and hand off to `@do`** (project orchestrator).

This skill is a **status supervisor only**. It does not recover jobs, resubmit, edit code, or redesign workflows.

Follow project rules: **slurm-execution-policy**, **validation-first**, **missing-data-policy**, **reproducibility**.

**Exclusive invoker under project execution:** only `@do` (step 4.3) launches `@monitor` during `/do`. `@prompt-orchestrator` must not invoke `@monitor` (avoids double monitoring).

---

## Monitoring duration inheritance

| Invoker | Duration |
|---------|----------|
| Direct (`@monitor` / user) | **30 minutes** (default) |
| `@do` (step 4.3) | Inherited from `@do` — default **2 hours** unless user override |
| Explicit user override (any invoker) | User-specified duration wins |

`@do` must pass the monitoring duration when launching `@monitor`. Do not silently fall back to 30 minutes when invoked by `@do`.

---

## Default behavior

- Apply duration inheritance above.
- Poll periodically (default: every **2–5 minutes**).
- Detect running, completed, failed, and stalled jobs.
- Estimate progress when workflow-specific indicators exist; otherwise **Unknown**.
- On error / unrecoverable stall → escalate to `@do` and end.

---

## Supervision checklist

```
Monitoring:
- [ ] Phase 1: Discover running tasks (+ progress indicators)
- [ ] Phase 2: Health + progress polling (loop until stop)
- [ ] On error → escalate to @do
- [ ] Final report: monitoring-report.md
```

---

## Phase 1 — Discover running tasks

Inspect active computational tasks for the current project (or the job set passed by `@do`):

- SLURM jobs
- Nextflow / Snakemake
- long-running local scripts / containers

### Discovery commands (use as applicable)

```bash
squeue -u "$USER" -o "%.18i %.30j %.8T %.10M %.6D %R"
ps aux | rg -i "snakemake|nextflow|python|Rscript|docker|apptainer"
ls -la data/logs/ logs/ 2>/dev/null
```

### Task registry

| Field | Source |
|-------|--------|
| execution environment | SLURM / local / workflow |
| job identifier | JobID, PID, session ID |
| start time | squeue, ps, logs |
| current state | RUNNING, FAILED, COMPLETED, … |
| expected outputs | sbatch / workflow / task file |
| log locations | `--output`, `--error`, workflow logs |
| progress indicators | workflow-specific (below) |

### Progress indicators (illustrative)

| Environment | Indicators |
|-------------|------------|
| Snakemake | Rules done / total |
| Nextflow | Completed processes / total |
| SLURM array | Tasks finished / array size |
| Downloads | Files or bytes done / expected |
| Training | Epoch / step vs max in logs |

If none exist, progress = **Unknown** — never invent percentages.

---

## Phase 2 — Health and progress monitoring

Until a stop condition:

**Health:** process state, exit status, log tails, output mtime/size, runtime vs wall time, scheduler state.

**Progress (when possible):** percent complete, ETA, stall warnings (indicator frozen for **2× poll interval** while supposedly RUNNING).

### Detect and record

- crashes, OOM, segfaults, timeouts
- stalled execution / stalled progress
- missing expected outputs
- scheduler failures
- dependency / path errors in logs
- ETA exceeding remaining SLURM wall time

Record timestamp + log excerpt for each event.

---

## Escalation (errors → `@do`)

On **any** job failure, unrecoverable stall, or condition that requires recovery / redesign:

1. Stop monitoring (do not retry, resubmit, or call `@debug` / `@code-review` / `@prompt-orchestrator`).
2. Hand off to **`@do`** (project orchestrator) with the escalation payload below.
3. Write `monitoring-report.md` including the escalation.

```markdown
## Monitor escalation → @do

**Task / job:** [name / ID]
**Failure type:** [OOM / timeout / crash / stall / missing output / …]
**State at stop:** [FAILED / TIMEOUT / STALLED / …]

### Logs
- stdout: [path]
- stderr: [path]
- excerpt: [error lines]

### Recommendation for @do
[Return to 4.1 with @debug / mark RECOVERABLE / user input needed]
```

---

## Stop conditions

| Condition | Action |
|-----------|--------|
| All watched jobs COMPLETED | End; success report |
| Duration elapsed | End; report still-running jobs |
| Failure / unrecoverable stall | End; escalate to `@do` |
| User interrupt | End; snapshot current state |

Use `Await` / periodic polling — do not busy-wait; note each poll for the final report.

---

## Final report

Generate **`monitoring-report.md`** (prefer `docs/monitoring-report.md`). Template: [monitoring-report-template.md](monitoring-report-template.md).

Include: monitored jobs, environments, job IDs, duration, progress estimates, detected failures, escalations to `@do`, completed jobs, recommendations.

---

## Execution rules

- Never terminate healthy jobs.
- Never resubmit, restart, or recover jobs.
- Never overwrite outputs or edit code.
- Never invoke `@prompt-orchestrator`, `@code-review`, or `@debug`.
- Preserve log files and failed partial outputs for `@do` / `@debug`.
- Never fabricate progress percentages.

## Coordination

| Skill / rule | When |
|--------------|------|
| `@do` | Sole invoker under `/do` (step 4.3); sole escalation target on error |
| `@prompt-orchestrator` | Must **not** invoke `@monitor` — `@do` owns monitoring |
| Rule **validation-first** | Apply when noting whether expected outputs appeared (report only; no repair) |

## Artifact registration

Register every generated report in `docs/artifact-registry.md` immediately after write (artifact, producer skill, generation date, purpose, status, downstream consumers).

Update existing rows when regenerating the same path; mark replaced paths `superseded`.

Format: [artifact-registry-template.md](../_shared/artifact-registry-template.md). Project rule: `artifact-registry` (alwaysApply).

## Additional resources

- Report template: [monitoring-report-template.md](monitoring-report-template.md)
