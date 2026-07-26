# Artifact Registry

**Path:** `docs/artifact-registry.md` (or project-root `artifact-registry.md`)

Living index of every generated report, graph, manifest, and checkpoint.
Skills append or update rows; they do not remove history — mark superseded entries.

## Entry format (table)

```markdown
# Artifact Registry

| artifact | producer skill | generation date | purpose | status | downstream consumers |
|----------|----------------|-----------------|---------|--------|----------------------|
| docs/dependency-graph.md | verify-todo | 2026-07-16 | Task dependency graph | final | prepare, do |
| docs/dependency-graph.mmd | verify-todo | 2026-07-16 | Mermaid dependency graph | final | prepare, do |
```

## Field definitions

| Field | Required | Values / notes |
|-------|----------|----------------|
| artifact | yes | Repo-relative path |
| producer skill | yes | Skill `name` from frontmatter |
| generation date | yes | YYYY-MM-DD |
| purpose | yes | One short phrase |
| status | yes | `draft` \| `final` \| `superseded` \| `failed` \| `active` |
| downstream consumers | yes | Comma-separated skills/tasks, or `—` |

## Update protocol

1. Write the artifact file.
2. Open `artifact-registry.md` (create from this template if missing).
3. If path already listed: update date, purpose, status, consumers; set prior status to `superseded` only when the *path* changes.
4. If new path: append a row.
5. Never leave a generated report/graph/manifest/checkpoint unregistered.

## Status meanings

| status | Meaning |
|--------|---------|
| draft | Intermediate; may change this session |
| final | Deliverable for current cycle |
| active | Living file under ongoing update (e.g. todo.md) |
| superseded | Replaced by a newer artifact path or version |
| failed | Produced to document failure (e.g. blocking dependency-report) |
