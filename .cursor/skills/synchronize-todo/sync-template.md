# Sync Summary Templates

## Sync Summary Report

```markdown
# Todo Sync Summary

**Sync date:** YYYY-MM-DD
**todo.md last updated:** YYYY-MM-DD
**Repository snapshot:** [branch / commit if available]

## Newly COMPLETED
| Task ID | Title | Evidence |
|---------|-------|----------|
| T-2.1 | Dataset audit | `docs/dataset-audit.md` — status Ready with warnings |
| T-1.1 | Environment setup | `environment.yml`, conda env verified |

## RUNNING (partial)
| Task ID | Title | Done | Remaining |
|---------|-------|------|-----------|
| T-3.1 | Taxonomic profiling | Snakemake rule exists; 80/100 samples in `results/kraken/` | 20 samples failed; rerun or exclude |

## New tasks added
| Task ID | Title | Reason |
|---------|-------|--------|
| T-3.3 | Rerun failed Kraken2 samples | 20 missing outputs detected; no task covered reruns |

## SKIPPED
| Task ID | Title | Reason |
|---------|-------|--------|
| T-4.0 | Legacy QIIME1 export | Superseded by T-4.1 Kraken2 workflow in `Snakefile` |

## Regressions / needs confirmation
| Task ID | Issue | Action |
|---------|-------|--------|
| T-2.2 | Marked complete but `metadata/samplesheet.tsv` still has orphan IDs | Confirm downgrade to RUNNING? |

## Remaining work (critical path)
| Task ID | Priority | Status | Blocks |
|---------|----------|--------|--------|
| T-3.2 | P0 | TODO | M3 |
| T-4.1 | P0 | RUNNING | M4 |

## Milestone status after sync
| ID | Milestone | Status |
|----|-----------|--------|
| M2 | Dataset audit passed | COMPLETED |
| M3 | Core analysis complete | RUNNING |

## Statistics
- Tasks COMPLETED this sync: N
- Tasks partial: N
- Tasks TODO/READY: N
- Tasks BLOCKED: N
- New tasks added: N
```

## Evidence Evaluation Matrix

Use internally before updating statuses:

```markdown
| Task ID | Expected output | Found | Done when met | Verdict |
|---------|-----------------|-------|---------------|---------|
| T-2.1 | docs/dataset-audit.md | Yes | Yes — Ready documented | Completed |
| T-3.1 | results/kraken/*.report | Partial | No — not all samples | Partial |
```

## Manual Notes Preservation

Protected regions — copy verbatim into updated todo.md:

```markdown
## Manual notes
<!-- manual -->
User decision: exclude batch 2024-03 runs until resequencing.
Do not auto-mark T-2.2 complete.
<!-- /manual -->
```

Also preserve per-task lines:
```markdown
  - Note: waiting on PI approval for exclusion list
  - User: low priority until November
```

## Sync Log Entry (optional append to todo.md)

```markdown
## Sync log
- **2026-07-15:** T-2.1 → COMPLETED; T-3.1 → RUNNING; added T-3.3
```

## Detection Heuristics

| Signal | Likely status |
|--------|----------------|
| All output files + Done when satisfied | Completed |
| Output dir exists but incomplete | Partial |
| Code committed, no outputs | RUNNING |
| Task references removed tool/path | Obsolete — verify before cancelling |
| New results/ dir with no todo task | Missing task — propose addition |
| Log shows FAILED | FAILED or RECOVERABLE — check Blocked by |

## Anti-Patterns

| Avoid | Prefer |
|-------|--------|
| Mark complete from empty placeholder files | Verify Done when criteria |
| Delete obsolete tasks | Status: SKIPPED + reason |
| Overwrite user notes | Append Synced:/Progress: lines |
| Recreate todo.md from scratch | Surgical field updates |
| Infer completion from git commit messages alone | Require artifact evidence |
