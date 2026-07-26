# Canonical task schema (`./todo/*.md`)

Every task file under `./todo/` MUST include a filled `<Task>` block (XML-like) plus optional human prose above it.

Do not omit fields — use `—` or `none` when empty.

## Template

```markdown
# T-2.1 — Dataset audit

## Description
[Free-form description from user prompt / prior notes]

## Task specification

<Task>
  <ID>T-2.1</ID>
  <Title>Dataset audit</Title>
  <Status>READY</Status>
  <Priority>P0</Priority>
  <Complexity>S</Complexity>
  <Objective>…</Objective>
  <Deliverables>docs/dataset-audit.md</Deliverables>
  <ExpectedOutputs>docs/dataset-audit.md</ExpectedOutputs>
  <AcceptanceCriteria>
    Audit status Ready or Ready with warnings; exclusion list finalized.
  </AcceptanceCriteria>
  <RequiredInputs>data/metadata/; data/raw/; data/manifests/</RequiredInputs>
  <Dependencies>T-1.1</Dependencies>
  <Dependents>T-2.2, T-3.1</Dependents>
  <Skills>dataset-auditor</Skills>
  <Rules>validation-first, missing-data-policy, task-status</Rules>
  <ExecutionEnvironment>local</ExecutionEnvironment>
  <ComputationalComplexity>Low</ComputationalComplexity>
  <EstimatedCPU>—</EstimatedCPU>
  <EstimatedMemory>—</EstimatedMemory>
  <EstimatedRuntime>—</EstimatedRuntime>
  <BlockedBy>—</BlockedBy>
  <Notes>—</Notes>
</Task>

## Execution metadata

| Field | Value |
|-------|-------|
| Task ID | T-2.1 |
| Skills | `@dataset-auditor` |
| Rules | validation-first, missing-data-policy |
| Inputs | `data/metadata/`, `data/raw/` |
| Outputs | `docs/dataset-audit.md` |
| Depends on | T-1.1 |
| Parallel with | — |
| Environment | local |
| Status | READY |
```

Keep the markdown **Execution metadata** table in sync with the `<Task>` block (for skills that parse tables).

## Field rules

| Field | Allowed values / notes |
|-------|------------------------|
| Status | `TODO` \| `READY` \| `RUNNING` \| `BLOCKED` \| `FAILED` \| `RECOVERABLE` \| `COMPLETED` \| `SKIPPED` only |
| Priority | `P0` \| `P1` \| `P2` |
| Complexity | `S` \| `M` \| `L` \| `XL` |
| Skills | Discovered skill `name` values; comma-separated; minimum set |
| Rules | Discovered rule stems; comma-separated |
| Dependencies / Dependents | Task IDs; `none` if empty |
| Estimated* | Fill when known; else `—` |

## Status selection guide

| Situation | Status |
|-----------|--------|
| New work, deps incomplete | `BLOCKED` or `TODO` |
| Deps COMPLETED, inputs available | `READY` |
| Out of scope / duplicate avoided by skipping | `SKIPPED` |
