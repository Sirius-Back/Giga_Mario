# Debug Report Template

## debug-report.md

Prefer `docs/debug-report.md`.

```markdown
# Debug Report

**Date:** YYYY-MM-DD
**Invoked by:** verify-todo | monitor | prompt-orchestrator | other
**Task / context:** [ID or job / path]
**Overall outcome:** Recovered | Escalated (Unsafe) | Escalated (Impossible) | Escalated (validation failed)

## Diagnosis

**Symptoms:**
- 

**Root cause:**
- 

**Evidence:**
- [log path / line, report excerpt, missing path]

**Cause class:** missing files | missing dependencies | incorrect paths | invalid task metadata | dependency graph inconsistencies | software configuration | environment activation | SLURM submission | resource allocation | workflow configuration | other

## Recovery planning

| Classification | Rationale |
|----------------|-----------|
| Safe \| Unsafe \| Impossible | |

## Attempted repairs

| Repair | Safe? | Result |
|--------|-------|--------|
| | yes/no | done / skipped / failed |

**Scientific outputs touched?** No (required) | [if yes — must escalate as failure]

## Validation

| Verifier skill | Result |
|----------------|--------|
| verify-todo \| project-auditor \| code-review \| other | pass / fail / not run |

## Remaining issues

- 

## Recommended manual actions

1. 
2. 

## Artifacts

| Path | Notes |
|------|-------|
| docs/debug-report.md | this report |
| | preserved logs / backups |
```
