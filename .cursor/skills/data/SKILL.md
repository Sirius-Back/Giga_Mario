---
name: data
description: >-
  Orchestrate dataset preparation by coordinating get-data, dataset-auditor,
  and project-auditor. Use when the user asks to prepare data, make the project
  data-ready, run the full data pipeline, or plan acquisitions without downloading.
disable-model-invocation: true
---

# Data

## Purpose

Prepare all datasets required for a scientific project by orchestrating the existing project skills. This skill is responsible for acquiring, validating, auditing and preparing datasets for downstream analyses. It should coordinate other skills instead of duplicating their functionality.

Follow project rules: **validation-first**, **reproducibility**, **missing-data-policy**, **scientific-integrity**.

## Subordinate skills

- `@get-data`
- `@dataset-auditor`
- `@project-auditor`

Read each subordinate skill before invoking. **Never duplicate** their workflows — delegate and summarize.

## Orchestration checklist

```
Data preparation:
- [ ] Phase 0: Detect current project state
- [ ] Phase 1: Data acquisition (@get-data) — if needed
- [ ] Phase 2: Dataset audit (@dataset-auditor)
- [ ] Phase 3: Scientific project audit (@project-auditor)
- [ ] Phase 4: Consolidated report (data-preparation-report.md)
```

---

## Phase 0 — Detect current project state

Before invoking any other skill:

- Inspect the project structure.
- Detect existing data directories.
- Inspect download manifests.
- Inspect checksum files.
- Inspect previous acquisition reports.
- Inspect previous dataset audit reports.
- Inspect previous scientific audit reports.
- Determine whether requested datasets are already present.
- Determine whether previous validation completed successfully.
- Detect partially completed downloads.
- Detect corrupted datasets.
- Detect missing datasets.
- Detect newly requested datasets.
- Detect obsolete or duplicated datasets.

**Key paths to inspect:**

| Artifact | Location |
|----------|----------|
| Raw data | `data/raw/` |
| Manifest | `data/manifests/download_manifest.tsv` |
| Checksums | `data/checksums/` |
| Acquisition report | `data/manifests/acquisition_report.md` |
| Dataset audit | `docs/dataset-audit.md` |
| Project audit | `docs/project-audit.md` |
| Download log | `data/logs/download.log` |

Build a **state table** per requested accession/dataset:

| ID | Present | Validated | Status | Action |
|----|---------|-----------|--------|--------|
| SRR123 | yes | pass | complete | skip download |
| SRR456 | partial | fail | corrupted | re-acquire via @get-data |

### Decision logic

- If all requested datasets are already present and successfully validated, **skip** `@get-data`.
- If only a subset of datasets is missing or failed validation, invoke `@get-data` **only for those datasets**.
- Never download datasets that have already passed validation unless explicitly requested.
- Preserve validated datasets whenever possible.

Record every skip decision with reason for Phase 4 report.

---

## Phase 1 — Data acquisition

Invoke `@get-data` when Phase 0 identifies missing, failed, or unvalidated required datasets.

Pass to `@get-data`:

- Only the subset requiring acquisition (not full list if others validated)
- User destination directory if specified
- Dry-run flag if in planning mode (see Optional mode below)

`@get-data` handles:

- discovering official repositories
- selecting the preferred download method
- downloading datasets
- checksum verification
- biological validation
- manifest generation
- download report generation

**If acquisition fails for any required dataset:**

- stop execution;
- report the reason;
- do not continue with downstream phases.

Distinguish **download failures** (network, auth, missing accession) from **validation failures** (checksum, biological) in reporting.

---

## Phase 2 — Dataset audit

Invoke `@dataset-auditor` after successful acquisition or when re-auditing existing validated data at user request.

Inspect every downloaded dataset for:

- metadata completeness
- missing samples
- duplicated samples
- sample identifier consistency
- sequencing layout
- paired-end consistency
- sequencing depth (when applicable)
- class balance
- missing values
- outliers
- batch effects
- repository consistency
- biological consistency

Generate a structured dataset quality report (`docs/dataset-audit.md` per subordinate skill).

**If critical issues are detected**, report them before continuing. For **Not ready** audit status, stop before declaring data-ready unless user explicitly overrides.

---

## Phase 3 — Scientific project audit

Invoke `@project-auditor` (read-only) to evaluate whether the project is ready for downstream scientific analyses.

Inspect:

- project structure
- reproducibility
- documentation
- manifests
- metadata
- download reports
- audit reports
- software organization
- data organization
- consistency between datasets and metadata
- missing files
- readiness for downstream workflows

Reference `docs/project-audit.md` in consolidated report. Do not duplicate full audit — summarize findings and P0/P1 items.

---

## Phase 4 — Consolidated report

Generate:

**`data-preparation-report.md`**

Save to `docs/data-preparation-report.md` or user path. Use [report-template.md](report-template.md).

Summarize:

- requested datasets
- repositories used
- acquisition methods
- downloaded datasets
- skipped datasets
- failed downloads
- biological validation summary
- dataset audit summary
- scientific audit summary
- detected issues
- unresolved problems
- recommended next steps

Do not replace reports generated by subordinate skills. Instead, **reference and summarize** them with paths.

---

## Execution rules

- This skill acts only as an orchestrator.
- Never duplicate functionality implemented by subordinate skills.
- Invoke subordinate skills only when required.
- Stop immediately if mandatory data acquisition fails.
- Clearly distinguish download failures from validation failures.
- Reuse existing validated datasets whenever possible.
- Preserve all generated manifests and reports.
- Preserve reproducibility across repeated executions.
- Record every skipped phase together with the reason.
- Never overwrite validated datasets unless explicitly requested.

---

## Optional mode — Dry Run

If the user requests planning only:

Do not modify the project.

Instead:

- determine which datasets would be downloaded;
- estimate download size;
- estimate temporary storage;
- estimate final storage;
- estimate download time;
- identify required repositories;
- identify required software;
- identify required databases;
- identify expected outputs;
- identify missing prerequisites.

Run Phase 0 fully. Invoke `@get-data` logic in plan-only mode (download plan without execution). Skip Phases 2–3 or run read-only audits if user requests.

Generate a complete download plan without downloading any files. Output: `docs/data-preparation-plan.md`.

---

## Success criteria

The project is considered **data-ready** only if:

- ✓ All requested datasets have been acquired or were already available.
- ✓ Technical validation has passed.
- ✓ Biological validation has passed.
- ✓ Dataset audit has completed successfully.
- ✓ Scientific audit reports no critical blocking issues.
- ✓ All reports and manifests have been generated.
- ✓ The project is ready for downstream computational analyses.

If any criterion fails, state which failed and what action unblocks it. Do not claim data-ready without evidence from subordinate reports.

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

- Consolidated report template: [report-template.md](report-template.md)
