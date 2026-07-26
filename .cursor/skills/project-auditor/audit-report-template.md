# Project Audit Report Template

```markdown
# Project Audit Report

**Date:** YYYY-MM-DD
**Repository:** [path or name]
**Branch / commit:** [if available]
**Auditor mode:** Read-only (no files modified)

## Executive summary
- [≤10 bullets: top findings and overall readiness]

## Overall scores
| Dimension | Rating | Summary |
|-----------|--------|---------|
| Project completeness | ☐ Poor ☐ Fair ☐ Good ☐ Excellent | |
| Documentation | ☐ Poor ☐ Fair ☐ Good ☐ Excellent | |
| Reproducibility | ☐ Poor ☐ Fair ☐ Good ☐ Excellent | |
| Methodological rigor | ☐ Poor ☐ Fair ☐ Good ☐ Excellent | |
| Publication readiness | ☐ Not ready ☐ Early ☐ Substantial ☐ Near-ready ☐ Ready | |

---

## 1. Project completeness
[What exists vs expected for this project type]

| Component | Status | Evidence |
|-----------|--------|----------|
| Raw data / manifests | Present / Partial / Missing | `path` |
| Processing workflow | ... | ... |
| Analysis outputs | ... | ... |
| Manuscript drafts | ... | ... |

## 2. Missing documentation
| Document | Gap | Impact | Priority |
|----------|-----|--------|----------|
| README | No run instructions | Users cannot reproduce | P0 |
| method-decision.md | Absent | Method rationale untracked | P1 |

## 3. Reproducibility issues
| Issue | Evidence | Impact | Priority |
|-------|----------|--------|----------|
| Unpinned dependencies | no version pins in environment.yml | Non-deterministic runs | P0 |
| Absolute paths | scripts/run.sh:12 `/home/user/...` | Fails on other machines | P1 |
| Missing random seeds | scripts/ml.R — no seed set | Irreproducible ML | P1 |

## 4. Inconsistent methods

**Source of truth:** `@verify-methods` — do not reconstruct methods in this audit.
**Artifacts cited:** `method-decision.md`, `docs/methods-verification.md` (or path from artifact-registry.md)

| Conflict | From verify-methods | Audit note | Priority |
|----------|---------------------|------------|----------|
| Classifier mismatch | SOTA / Decision vs docs | README still lists MetaPhlAn | P0 |

## 5. Outdated software
| Tool | Pinned version | Concern | Evidence basis | Priority |
|------|----------------|---------|----------------|----------|
| tool X | v1.0 | Superseded / security | env file; literature Uncertain unless verified | P2 |

*Do not claim outdated without evidence or mark Uncertain.*

## 6. Duplicated functionality
| Function | Locations | Recommendation |
|----------|-----------|----------------|
| FASTQ QC wrapper | scripts/qc.sh, bin/qc.py | Consolidate to single module | P2 |

## 7. Incomplete analyses
| Analysis | State | Evidence | Priority |
|----------|-------|----------|----------|
| Differential abundance | Partial | results/de/ — 80/100 samples | P0 |
| Beta diversity PERMANOVA | Not started | no script/output | P1 |

## 8. Code quality notes
| Area | Finding | Location | Priority |
|------|---------|----------|----------|
| Silent exception | bare `except:` | scripts/x.py:45 | P1 |
| No input validation | missing file check | scripts/y.py | P1 |

## 9. Statistical methodology

Summarize from `@verify-methods` report; do not re-derive tests independently.

| Check | Pass/Fail | Cite verify-methods / artifact |
|-------|-----------|--------------------------------|
| Multiple testing correction | … | … |
| Effect sizes reported | … | … |
| Test matches design | … | … |

## 10. Figures and outputs
| Check | Status | Notes |
|-------|--------|-------|
| Vector figures | Partial | PNG only in figures/ |
| Colorblind-safe palette | Unknown | no plot_style config |
| Figure–Results alignment | Gap | no figure plan doc |

## 11. Tracking artifacts
| File | Status | Notes |
|------|--------|-------|
| todo.md | Out of sync | T-3.1 marked TODO; outputs exist |
| method-decision.md | Incomplete | Kraken2 choice not logged |

## 12. Publication readiness
**Rating:** [Not ready | Early | Substantial | Near-ready | Ready]

**Rationale:** [Evidence-based paragraph]

**Blocking items for submission:**
1. ...
2. ...

---

## Recommendations (prioritized)

### P0 — Critical
| # | Recommendation | Impact | Effort |
|---|--------------|--------|--------|
| 1 | Fix metadata ↔ FASTQ ID mismatches | Invalid joins invalidate all downstream | M |

### P1 — High
| # | Recommendation | Impact | Effort |
|---|--------------|--------|--------|

### P2 — Medium
...

### P3 — Low
...

---

## Audit limitations
- [What was not inspected: external data, HPC job history, private docs]
- [Assumptions made]
- [Suggested follow-up audits or skills to run]
```

## Publication Readiness Rubric

| Rating | Typical evidence |
|--------|------------------|
| **Not ready** | Missing core analysis, broken reproducibility, no docs |
| **Early** | Pipeline runs but incomplete outputs; Methods not draftable |
| **Substantial** | Core results exist; major doc/stat gaps remain |
| **Near-ready** | Methods/Results draftable; minor figure/repro fixes |
| **Ready** | Verified reproducibility, complete analyses, manuscript artifacts polished |

Rating must cite specific evidence — not subjective impression.

## Completeness Checklist (scientific computing)

- [ ] README with install and run
- [ ] Environment lockfile or container
- [ ] Config-driven workflow
- [ ] Sample metadata validated
- [ ] QC before primary analysis
- [ ] method-decision.md maintained
- [ ] todo.md reflects progress
- [ ] Results outputs match plan
- [ ] Statistical reporting meets project rules
- [ ] Figures publication-ready or figure plan exists

## Finding Format (internal)

Use consistently before writing report:

```
ID: AUD-001
Category: Reproducibility
Priority: P0
Issue: [one sentence]
Evidence: [path:line or absence]
Impact: [why it matters]
Action: [specific fix — report only unless user requests edit]
```

## Anti-Patterns

| Avoid | Prefer |
|-------|--------|
| Modifying files during audit | Read-only report |
| Generic "improve documentation" | Name missing doc and section |
| Invented outdated tool claims | Cite env pin + Uncertain if no literature |
| Score without evidence | Rubric + paths |
| Duplicate full sub-skill reports | Unified summary with cross-refs |
