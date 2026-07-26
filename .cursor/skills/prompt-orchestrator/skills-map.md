# Skill Discovery Guide

Guide for `@prompt-orchestrator` (**single assigned task only** — not a project scheduler; `@do` owns waves/graph/order). **Not an exhaustive catalog.** Skills must be discovered at runtime from the project registry.

## Principle

Remove every hardcoded list of project skills from orchestration logic.

Instead, require the orchestrator to **dynamically discover** every installed project skill before execution.

Before invoking subordinate skills, determine whether the requested task is governed primarily by:

- project rules;
- project skills;
- both.

Apply all relevant rules automatically.

Invoke only the minimum necessary skills.

When selecting subordinate skills:

- inspect the project skill registry;
- discover every available skill;
- infer each skill's responsibility from its metadata;
- select the minimum necessary subset of skills;
- never assume that the built-in examples are exhaustive.

The examples shown in this documentation are **illustrative only** and must **never** limit skill discovery.

---

## Governance classification (illustrative)

| Request shape | Likely governance | Skills? |
|---------------|-------------------|---------|
| “Follow Nature style; don’t invent DOIs” | Rules | No |
| “Download these SRA runs” | Both | Yes — acquire/data skill(s) |
| “Run the prepared TODO waves” | Both | Yes — execution/prepare skills if present |
| “What SLURM resources should this use?” | Rules (+ maybe plan) | Often no |

Always re-evaluate from discovered registries — this table does not bind selection.

---

## Project skill registry

| Location | What counts as a skill |
|----------|------------------------|
| `.cursor/skills/*/SKILL.md` | Installed project skill |

### Discovery algorithm

1. List directories under `.cursor/skills/`.
2. For each directory with a `SKILL.md`, parse YAML frontmatter (`name`, `description`).
3. Optionally skim purpose / coordination sections to refine matching.
4. Build registry: `{ name, path, description, responsibility_summary }`.
5. Match user request + plan steps to registry entries by semantic fit to `description` / purpose.
6. Select the **smallest** set that covers required work without overlapping responsibilities.
7. Read each selected skill's full `SKILL.md` before invoking.
8. Record full registry + selection rationale in `docs/execution/<task-id>.md` (never `docs/execution-report.md`).

### Do not

- Skip discovery because a familiar skill name appears in examples below
- Treat absence from this file as “skill does not exist”
- Treat presence in examples as “must use this skill”
- Hardcode a closed list of skill names in plans or prompts

---

## Project rules

Discover under `.cursor/rules/*.mdc` (frontmatter `description`, `alwaysApply`).

Typical domains (illustrative — verify on disk): scientific integrity, reproducibility, statistics, figures, writing style, missing data, SLURM, validation-first, method-decision tracking.

---

## Illustrative matching patterns (non-exhaustive)

These examples show *how* to match metadata to intent. **Any installed skill** whose description fits better MUST be preferred over these names if they differ.

| User intent (example) | Selection heuristic |
|-----------------------|---------------------|
| Acquire public sequencing / repository data | Skill whose description covers download / accession / repository acquisition |
| Audit sample metadata and files before analysis | Skill covering dataset audit / QC / readiness |
| End-to-end data prep (acquire + audit + readiness) | Orchestrator-style data skill if present; else compose discovered acquire + audit skills |
| Draft Methods / Results / figures / rebuttals | Skills whose descriptions match manuscript section or figure planning |
| Validate TODO dependency graph | Skill that owns dependency-graph construction/validation |
| Plan then execute TODO waves | Skills for prepare / multi-agent execution if present |
| Turn a free-form request into todo tasks | Skill that transforms prompts into ./todo/ specifications |
| Turn architecture into a living `todo.md` | Skill that owns top-level project plan generation |
| Turn a free-form request into `./todo/*.md` specs | Skill that creates executable task files linked into `todo.md` |
| Supervise SLURM / long jobs (status only) | Skill whose description covers job status monitoring (invoked by project executor, not single-task orchestrator) |
| Independent implementation review | Skill describing code review / correctness vs spec |

### Illustrative pipelines (non-binding)

```
# Shape only — substitute whatever skills discovery returns
[data-related skills] → [methods verification] → [architecture viz] → [todo generation]

[methods verification] → [methods + results writers in parallel] → [figure design]

[todo sync] → [project audit] → [reviewer response]

[benchmark design] → compute (SLURM) → [monitor] → [results writer]

[architecture → main todo.md] → [./todo/*.md specs] → [dependency verify] → [prepare] → [do / execution engine]
# under do: per-task orchestrator → code-review → monitor (status) → auditor
```

Actual skill names and availability come **only** from Phase 0 discovery.

---

## Environment selection guide

| Task type | Environment |
|-----------|-------------|
| File inspection, reports | Local |
| Small script (< few GB, minutes) | Local Conda |
| Pipeline stage | Snakemake/Nextflow if defined |
| Large download / assembly / training | SLURM sbatch |
| Containerized tool | Docker/Apptainer if project provides |
| Continuous job supervision | Discovered monitor-like skill when present |

---

## Delegation anti-patterns

| Avoid | Prefer |
|-------|--------|
| Doing specialized work inline | Discovered skill whose metadata matches |
| Using only skills remembered from this file | Full registry scan every run |
| Selecting every vaguely related skill | Minimum necessary subset |
| Rebuilding dependency graphs if a graph-owner skill exists | Delegate to that skill |
| Auto-editing code when a read-only review skill exists | Review first; user-approved fixes |

---

## Progress artifacts (discover + update)

Do not assume a fixed writer. Update artifacts that the **selected** skills declare as outputs, and **register every generated file** in `artifact-registry.md` / `docs/artifact-registry.md`.

Common paths (illustrative): `todo.md`, `method-decision.md`, `docs/execution/<task-id>.md`, `docs/dependency-graph.md`, `data/manifests/*`.

Always write `docs/execution/<task-id>.md` from this orchestrator (Phase 8), including the discovered registry snapshot. **Never** generate or overwrite `docs/execution-report.md` (`@do` only). Register every report in `artifact-registry.md`.
