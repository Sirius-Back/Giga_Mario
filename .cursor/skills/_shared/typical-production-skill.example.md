---
name: example-leaf-skill
description: >-
  EXAMPLE ONLY — not an installed skill. Copy into .cursor/skills/<name>/SKILL.md
  when authoring a production leaf skill (one deliverable class: acquire, transform,
  audit, or write). Replace name/description/paths; delete this file after copying.
disable-model-invocation: true
---

# Example Leaf Skill (typical production pattern)

Modelled on `@get-data`: one job, fail-early validation, SLURM when heavy, verify
outputs, report + artifact registry. Not an orchestrator.

## Purpose

Produce one class of validated deliverables from explicit inputs.

Follow project rules by **name** (rules are not skills): **validation-first**,
**missing-data-policy**, **scientific-integrity**, **reproducibility**,
**artifact-registry**, plus domain rules as needed (**slurm-execution-policy**,
**statistical-analysis**, **nature-writing-style**, …).

## Inputs

| Input | Required | Notes |
|-------|----------|-------|
| User request or `./todo/<id>.md` | yes | IDs, paths, acceptance criteria |
| Metadata / config | if needed | Fail if missing — never invent |
| Prior artifacts | optional | Only paths that exist and validate |

## Pre-flight

1. Paths exist, readable, non-empty.
2. Required metadata columns/keys present.
3. Tools/versions checked when computation is involved.
4. If anything required is missing → stop; report what is missing (missing-data-policy).

## Workflow

```
example-leaf-skill:
- [ ] Step 1: Parse and validate inputs
- [ ] Step 2: Plan (size, resources, commands) — confirm if large
- [ ] Step 3: Execute (local or SLURM)
- [ ] Step 4: Verify outputs
- [ ] Step 5: Report + register artifacts
```

### Step 1–2

Prefer official sources/APIs. Record every command and software version.
Present a plan for user confirmation when cost/size exceeds a stated threshold.

### Step 3

Heavy work → sbatch with `--cpus-per-task`, `--mem`, `--time`, `--output`, `--error`.
Even CPU counts (16/32; 64 only when justified). Pass `${SLURM_CPUS_PER_TASK}` into tools.
Do not supervise long jobs yourself — under `/do`, `@monitor` (via `@do` 4.3) owns status checks.

### Step 4

Never assume success. Check dimensions, ID overlap, empty files, checksums/schema.
On failure: report paths and errors; do not fabricate results or mark COMPLETED.

### Step 5

Write `docs/<skill>-report.md` (or sibling `*-template.md`). Register every artifact.

## Deliverables

| Output | Path |
|--------|------|
| Primary product | `results/<skill>/…` or `data/…` |
| Report | `docs/<skill>-report.md` |
| Logs / manifests | `data/logs/`, `data/manifests/` |

## Coordination

| Skill / owner | Role |
|---------------|------|
| `@prompt-orchestrator` / `@do` / `@data` | May invoke this leaf; do not call them back |
| `@verify-todo` | Do not rebuild the dependency graph |
| `@monitor` | Do not invoke — status supervision is `@do` only under `/do` |

**Do not:** schedule other TODO tasks; overwrite `docs/execution-report.md` (`@do` only);
author main `todo.md` (`@generate-todo`) or `./todo/*.md` (`@prepare-prompt`) unless that is this skill’s purpose.

## Artifact registration

Append/update `docs/artifact-registry.md` immediately after each write:

| Field | Content |
|-------|---------|
| artifact | Repo-relative path |
| producer skill | This skill’s `name` |
| generation date | YYYY-MM-DD |
| purpose | One phrase |
| status | `draft` / `final` / `superseded` / `failed` / `active` |
| downstream consumers | Skills, tasks, or humans |

## Hard rules

- Never invent data, statistics, DOIs, or “successful” validation.
- Never silently skip failed steps.
- Cite rules by name; never list a rule as `@skill-name`.
- Prefer update of living docs over recreate.
- Keep `SKILL.md` under ~500 lines; put long material in `reference.md`.
