---
name: visualize-architecture
description: >-
  Generate an accurate Mermaid workflow diagram from the repository and
  method-decision.md showing data objects, transformations, and software.
  Use when the user asks for a pipeline diagram, workflow visualization,
  architecture map, or data-flow chart of the project.
disable-model-invocation: true
---

# Visualize Architecture

Inspect the complete project repository together with method-decision.md and automatically generate a comprehensive Mermaid workflow diagram.

Nodes should represent biological data, intermediate files, reference databases, metadata, statistical objects, machine learning models or final outputs. Prefer generic scientific object names such as FASTQ, BAM, MAG, Gene Counts, Protein FAA, Taxonomic Profiles, Differential Expression Table, etc.

Edges should describe transformations performed between objects, labelled with the corresponding software, algorithms, statistical analyses or workflow stages, such as FastQC, Trimmomatic, Kraken2, DESeq2, MAFFT, IQ-TREE, XGBoost, PCA, PERMANOVA, Random Forest, etc.

The generated diagram should accurately reflect the **implemented workflow** rather than an idealized pipeline. Infer missing links only when strongly supported by the project structure. Report uncertainty whenever workflow reconstruction is incomplete.

Follow project rules: **scientific-integrity**, **missing-data-policy**, **method-decision-tracking**, **validation-first**.

## Workflow

Copy and track progress:

```
Architecture visualization:
- [ ] Step 1: Map workflow artifacts
- [ ] Step 2: Extract data objects and transformations
- [ ] Step 3: Read method-decision.md
- [ ] Step 4: Build evidence-graded graph
- [ ] Step 5: Render Mermaid diagram
- [ ] Step 6: Write uncertainty report
```

### Step 1: Map workflow artifacts

Inspect:

| Source | Extract |
|--------|---------|
| Snakemake / Nextflow / WDL / CWL | Rule/process inputs, outputs, tool commands |
| Shell / Python / R scripts | Read/write paths, CLI tools invoked |
| Notebooks | Data load → transform → save chains |
| Config / sample sheets | File types, reference paths |
| `results/`, `data/`, `figures/` | Actual output types on disk |
| `method-decision.md` | Locked tools and parameters |
| README / docs | Stated pipeline (secondary to code) |

Prefer **executed workflow code** over documentation when they conflict.

### Step 2: Extract data objects and transformations

**Nodes** — use generic scientific object names:

| Category | Examples |
|----------|----------|
| Raw inputs | FASTQ, FASTA, VCF, Phenotype Table |
| Alignments | BAM, CRAM, SAM |
| Assemblies | Contigs, MAG, Scaffolds |
| Annotations | GFF, Gene Counts, Protein FAA |
| Profiles | Taxonomic Profiles, KO Abundance |
| References | Reference Genome, Kraken Index, Protein DB |
| Metadata | Sample Metadata, Batch Covariates |
| Statistical | Count Matrix, Normalized Matrix, Distance Matrix |
| ML | Trained Model, Predictions, Feature Importance |
| Outputs | Differential Expression Table, Figures, Report |

Use subgraphs for phases: `QC`, `Assembly`, `Annotation`, `Statistics`, `ML`.

**Edges** — label with tool or analysis name:

```
FASTQ -->|FastQC| QC Report
FASTQ -->|Trimmomatic| Trimmed FASTQ
Trimmed FASTQ -->|Kraken2| Taxonomic Profiles
```

Include version in label only when verified in env/config (e.g., `Kraken2 v2.1.3`).

### Step 3: Read method-decision.md

- **Locked** entries → solid edges with stated tools
- **Tentative / Open** → dashed edges or `-.->` in Mermaid
- Contradictions between code and method-decision.md → note in uncertainty report

### Step 4: Build evidence-graded graph

Assign each node and edge:

| Grade | Meaning | Diagram treatment |
|-------|---------|-------------------|
| **Verified** | Explicit in workflow code or config I/O | Solid node/edge |
| **Inferred** | Strong structural support (path convention, rule name) | Dashed edge; suffix `(inferred)` |
| **Unknown** | Cannot connect | Omit from diagram; list in report |

Infer missing links **only when strongly supported** — e.g., Snakemake rule output filename matches next rule input pattern. Do not connect stages absent from repo unless inference is documented.

### Step 5: Render Mermaid diagram

Use conventions in [mermaid-template.md](mermaid-template.md).

Output:

1. **Main workflow diagram** — end-to-end data flow
2. **Optional phase diagrams** — if main graph exceeds ~40 nodes, split by subgraph

Save to user path or suggest `docs/workflow-architecture.md` (Mermaid fenced block + legend).

Use `flowchart TD` or `flowchart LR` based on pipeline depth; left-to-right for linear pipelines, top-down for branching.

Node shape hints:

- `([Metadata])` stadium for metadata
- `[[Reference DB]]` for databases
- `(Statistical Object)` rounded for derived tables
- `[FASTQ]` default rectangle for files

### Step 6: Write uncertainty report

Always include when any Verified edge is missing or Inferred edges exist:

- Unconnected inputs/outputs found on disk
- Rules/scripts with unclear I/O
- Conflicts between docs and code
- What evidence would resolve each gap

Deliver as section below diagram or `docs/workflow-architecture-gaps.md`.

## Deliverables

1. **Mermaid diagram** — implemented workflow
2. **Legend** — node types, solid vs dashed edges
3. **Uncertainty report** — incomplete reconstruction items

## Coordination

| Skill | Role |
|-------|------|
| `@verify-methods` | Method names and parameters for edge labels |
| `@methods-writer` | Cross-check diagram against Methods text |
| `@generate-todo` | Optional phase labels from project plan |

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

- Mermaid syntax, shapes, and examples: [mermaid-template.md](mermaid-template.md)
