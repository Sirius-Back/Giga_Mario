# Monitoring Report Template

## monitoring-report.md

```markdown
# Monitoring Report

**Date:** YYYY-MM-DD
**Monitoring start:** HH:MM
**Monitoring end:** HH:MM
**Duration:** 30 min (direct default) | 2 h (inherited from @do default) | [user-specified]
**Invoked by:** direct | do | other
**Duration source:** direct default | inherited from do | user override
**Poll interval:** ~N min
**Overall status:** All complete | Partial | Unrecoverable failure | Period expired

## Executive summary
[3–5 sentences: jobs watched, failures, recoveries, current state]

---

## Monitored jobs registry

| Job ID / PID | Name | Environment | State (start → end) | Resources | Start | End |
|--------------|------|-------------|---------------------|-----------|-------|-----|
| 12345 | megahit_asm | SLURM | RUNNING → COMPLETED | 32 CPU, 128G | 08:00 | 08:45 |
| 12346 | get_data | SLURM | FAILED → RUNNING (12401) | 32 CPU, 64G | 08:10 | — |

### Log locations
| Job | stdout | stderr |
|-----|--------|--------|
| 12345 | data/logs/megahit_12345.out | data/logs/megahit_12345.err |

### Expected outputs
| Job | Expected | Observed |
|-----|----------|----------|
| 12345 | results/megahit/sample/contigs.fa | present, 12 MB |

### Progress estimates
| Job | Indicator | % complete | Remaining runtime | ETA (completion) | Confidence | Stall warning |
|-----|-----------|------------|-------------------|------------------|------------|---------------|
| 12345 | samples 40/100 | 40% | ~90 min | 2026-07-16 10:15 UTC | Approximate | no |
| 12346 | — | Unknown | Unknown | Unknown | Unknown | yes — no progress 10 min |

---

## Health monitoring timeline

| Time | Event | Job | Details |
|------|-------|-----|---------|
| 08:15 | poll | 12345 | RUNNING; 35%→40%; ETA 10:15 UTC |
| 08:22 | stall warning | 12346 | progress frozen 10 min; still RUNNING |
| 08:23 | failure | 12346 | OOM in .err — Killed |

---

## Detected failures

| Time | Job | Type | Root cause | Log excerpt |
|------|-----|------|------------|-------------|
| 08:23 | 12346 | OOM | 64G insufficient | `slurmstepd: ... Killed` |

---

## Recovery actions

| Time | Job | Attempt | Action | New Job ID | Outcome |
|------|-----|---------|--------|------------|---------|
| 08:25 | 12346 | 1 | Resubmit --mem=128G | 12401 | recovered |

**Recovery limit:** 2 attempts per task

### Unrecoverable failures

| Job | Attempts | Reason | User action required |
|-----|----------|--------|------------------------|
| — | — | — | — |

---

## Code review summary (Phase 4 — @code-review)

**Report:** docs/code-review.md | inline

| ID | Severity | Issue | Monitor action |
|----|----------|-------|----------------|
| CR-001 | Critical | Missing input validation | No retry — user fix required |

**Operational fixes applied:** resubmit 128G mem (attempt 1)

---

## Escalation (@prompt-orchestrator)

**Escalated:** Yes | No

| Field | Value |
|-------|-------|
| Reason | Max recovery attempts / architectural issue |
| Handoff time | HH:MM |
| Orchestrator action | pending |

[Include full escalation payload if escalated]

---

## Post-recovery validation

| Job | Restart OK | Outputs generating | Error cleared | Extended monitor |
|-----|------------|-------------------|---------------|------------------|
| 12401 | yes | yes | yes | +30 min from 08:26 |

---

## Completed jobs

| Job ID | Name | Exit code | Outputs valid | Completed at |
|--------|------|-----------|---------------|--------------|
| 12345 | megahit_asm | 0 | yes | 08:45 |

---

## Still running at report time

| Job ID | Name | State | Notes |
|--------|------|-------|-------|
| 12401 | get_data | RUNNING | monitoring period expired |

---

## Resource and disk notes

| Metric | Observation |
|--------|-------------|
| Disk usage | 78% on /mnt/tank/scratch |
| SLURM queue | 12401 RUNNING |

---

## method-decision.md updates

| Decision | Change | Reason |
|----------|--------|--------|
| get-data SLURM memory | 64G → 128G | OOM on SRR2345678 |

---

## Recommendations

1. [Next action — e.g., run @synchronize-todo after 12401 completes]
2. [Preventive — e.g., default get-data sbatch to 128G for large accessions]

---

## Execution rules compliance

- [x] Healthy jobs not terminated
- [x] Completed jobs not resubmitted
- [x] Successful outputs preserved
- [x] Failed logs preserved
```

## Stalled job heuristics

| Signal | Threshold | Likely cause |
|--------|-----------|--------------|
| No log growth | 2× poll interval | Stuck I/O, deadlock |
| Progress % / count unchanged | 2× poll interval while RUNNING | Stalled progress |
| RUNNING but 0% CPU long period | scheduler / wait | Dependency wait |
| Output size unchanged | 2× poll interval | Hung process |
| ETA > remaining wall time | any poll | Likely TIMEOUT |
| PD indefinitely | queue policy | Priority, QOS |

## SLURM state reference

| State | Meaning |
|-------|---------|
| RUNNING | Active |
| COMPLETED | Finished success |
| FAILED | Non-zero exit |
| TIMEOUT | Wall time exceeded |
| OUT_OF_MEMORY | OOM kill |
| CANCELLED | User/system cancel |
| PD | Pending |

## Poll cycle checklist (internal)

Each poll:
1. Refresh job registry (new + completed since last poll)
2. Tail logs for errors since last offset
3. Check expected output mtimes/sizes
4. Update workflow-specific progress indicator → % complete, remaining runtime, ETA
5. Evaluate health + **stalled progress** heuristics; emit stall warnings
6. Trigger Phase 3–5 if failure detected
7. Log snapshot (including progress) for final report

## Minimal report (no failures)

When all jobs complete successfully with no recovery:

- Executive summary
- Monitored jobs registry
- Progress estimates (final 100% / completed)
- Completed jobs
- Recommendations (optional)
