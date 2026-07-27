---
name: project-auditor
description: >-
  Perform a comprehensive scientific audit of the repository covering
  reproducibility, documentation, and publication readiness. Delegates
  methodological reconstruction to verify-methods (single source of truth).
  Use when the user asks for a project audit, health check, or publication
  readiness assessment.
disable-model-invocation: true
---

# Project Auditor

Perform a comprehensive scientific audit & code review of the entire repository.

Inspect project structure, documentation, workflows, code quality, reproducibility, statistical methodology, figures, method-decision.md, todo.md, software organization and generated outputs.

Produce a structured report containing:

- project completeness
- missing documentation
- reproducibility issues
- inconsistent methods
- outdated software
- duplicated functionality
- incomplete analyses
- publication readiness
- recommendations prioritized by impact

Do not modify project files unless explicitly requested.

Follow all project rules. This skill is **read-only** by default — inspect and report only.

## Workflow

Copy and track progress:

```
Project audit:
- [ ] Step 1: Repository inventory
- [ ] Step 2: Documentation and tracking review
- [ ] Step 3: Workflow and code quality review
- [ ] Step 4: Scientific methodology review → invoke @verify-methods
- [ ] Step 5: Outputs and publication readiness
- [ ] Step 6: Synthesize findings and prioritize
- [ ] Step 7: Write audit report
```

### Step 1: Repository inventory

Map top-level structure and purpose:

| Area | Inspect |
|------|---------|
| Layout | `data/`, `src/`, `config/`, `results/`, `figures/`, `docs/`, `tests/` |
| Entry points | README, Snakefile, Makefile, main scripts |
| Environment | `environment.yml`, Docker, `requirements.txt`, lockfiles |
| Version control | `.gitignore`, large binaries tracked, secrets exposure |
| HPC | SLURM scripts, log directories |

Note what exists vs what a reproducible scientific project typically requires.

### Step 2: Documentation and tracking review

| Artifact | Checks |
|----------|--------|
| README | Install, run instructions, data requirements |
| `method-decision.md` | Present, entries with evidence, Locked vs Tentative |
| `todo.md` | Present, statuses vs repo reality (cross-check like `@synchronize-todo` read-only) |
| Methods/Results drafts | `docs/` completeness |
| Architecture diagram | Mermaid or workflow doc present |

Flag **missing documentation** with specific gaps (what doc, what section, why it matters).

### Step 3: Workflow and code quality review

Inspect workflows, scripts, notebooks:

- **Modularity** — duplicated logic across scripts
- **Validation** — input checks before processing (**validation-first**)
- **Error handling** — no silent failures or swallowed exceptions
- **Paths** — relative vs hard-coded absolute paths
- **Secrets** — credentials, tokens in code
- **Tests** — presence and coverage of critical functions
- **Style** — consistent naming, reasonable complexity

Flag **duplicated functionality** with file paths for each duplicate.

Do not refactor code — report only.

### Step 4: Scientific methodology review

Whenever methodological consistency is audited, **invoke `@verify-methods`** instead of independently reconstructing scientific methods.

`@verify-methods` remains the **single source of truth** for methodological reconstruction (decisions, parameters, SOTA comparison, `method-decision.md` updates per that skill’s rules).

**Do not:**

- re-derive methods from code/notebooks in parallel to `@verify-methods`
- rebuild a separate method inventory
- invent SOTA ratings without `@verify-methods` output

**Do:**

1. Invoke `@verify-methods` (read its `SKILL.md` and follow it).
2. Consume its deliverables: updated/verified `method-decision.md`, `docs/methods-verification.md` (or registered path in `artifact-registry.md`).
3. Summarize methodological findings in the project audit by **citing** those artifacts.
4. Add only audit-layer checks that are *not* method reconstruction:

| Check | Source after verify-methods |
|-------|-----------------------------|
| Method consistency (audit) | Align workflow/docs claims with `@verify-methods` report — flag mismatches as audit findings |
| Statistical reporting in outputs | Presence of FDR/effect sizes in result tables (not re-choosing tests) |
| QC gates | Whether dataset-audit / QC artifacts exist |
| Benchmark rigor | Whether benchmark protocol artifacts exist |
| Incomplete analyses | Orphan results / missing downstream steps |

If `@verify-methods` is unavailable, stop the methodology subsection, report the gap, and do not reconstruct methods independently.

Flag **inconsistent methods** and **incomplete analyses** only from verify-methods outputs plus non-reconstructive repo evidence (paths, missing files).

### Step 5: Outputs and publication readiness

| Check | Criteria |
|-------|----------|
| Results completeness | Expected outputs from workflow vs `results/`, `figures/` |
| Figure quality | Vector export, palettes, labels (**publication-figures**) |
| Manuscript artifacts | Methods, Results, figure plan drafts |
| Reproducibility bundle | Env file + config + entry command reruns pipeline |
| Provenance | Logs, versions, intermediate files where needed |

Score **publication readiness** (see template): Not ready / Early / Substantial / Near-ready / Ready — with evidence, not opinion alone.

### Step 6: Synthesize findings and prioritize

Assign each finding:

| Priority | Definition |
|----------|------------|
| **P0 — Critical** | Blocks correctness, reproducibility, or publication integrity |
| **P1 — High** | Major gap; should fix before submission or release |
| **P2 — Medium** | Improves quality; not blocking |
| **P3 — Low** | Polish, optional enhancements |

Each recommendation: **Issue → Evidence → Impact → Recommended action**.

Never invent issues — every finding needs a repo path or explicit absence (e.g., "no environment.yml found").

### Step 7: Write audit report

Use [audit-report-template.md](audit-report-template.md).

Save to user path or suggest `docs/project-audit.md`. Do **not** write to project files unless user explicitly requests fixes or updates.

Optional deliverable: executive summary (≤10 bullets) at top for PI/lab meeting.

## Scope boundaries

**In scope:** Read-only inspection of entire repo; structured report.

**Out of scope unless requested:**
- Editing code, configs, `todo.md`, or `method-decision.md`
- Running pipelines or destructive commands
- Auto-fixing findings

When user asks to fix issues, delegate to appropriate skills (discover from `.cursor/skills/`; methodology fixes go through `@verify-methods` first) with explicit permission.

## Coordination with other skills

| Skill | Audit dimension |
|-------|-----------------|
| `@verify-methods` | **Sole** methodological reconstruction & SOTA; project-auditor consumes only |
| `@dataset-auditor` | Data readiness |
| `@synchronize-todo` | Plan vs progress |
| `@visualize-architecture` | Workflow map completeness |
| `@methods-writer` / `@results-writer` | Manuscript readiness |
| `@figure-designer` | Figure plan vs outputs |
| `@reproducibility` rule | Env, seeds, paths |

Produce one unified project-audit report that **references** verify-methods artifacts rather than duplicating method reconstruction.

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

- Report template and scoring rubrics: [audit-report-template.md](audit-report-template.md)
