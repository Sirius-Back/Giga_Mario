# Mermaid Architecture Templates

## Main Diagram Template

```mermaid
flowchart LR
    subgraph Inputs
        META([Sample Metadata])
        REF[[Kraken2 PlusPF Index]]
        FQ[FASTQ]
    end

    subgraph QC
        FQ -->|FastQC| QC_RPT[QC Report]
        FQ -->|Trimmomatic| TRIM[Trimmed FASTQ]
    end

    subgraph Classification
        TRIM -->|Kraken2| TAX[Taxonomic Profiles]
    end

    subgraph Statistics
        TAX -->|Normalization| NORM[Normalized Matrix]
        NORM -->|Wilcoxon + BH-FDR| DE[Differential Abundance Table]
        NORM -->|PCA| PCO[PCoA Coordinates]
        PCO -->|PERMANOVA| PERM[PERMANOVA Results]
    end

    META -.->|join sample_id| TAX
    META -.->|group labels| DE

    classDef inferred stroke-dasharray: 5 5
```

## Edge Style Conventions

| Evidence | Mermaid syntax | Label suffix |
|----------|----------------|--------------|
| Verified | `-->|Tool|` | none |
| Inferred | `-.->|Tool (inferred)|` | `(inferred)` |
| Tentative method | `-.->|Tool (?)|` | from method-decision Open |

## Node Naming Rules

- Use **generic scientific names**, not raw filenames: `FASTQ` not `sample_R1.fq.gz`
- Filenames go in uncertainty report or footnote, not node labels
- Plural for sample-level collections: `Taxonomic Profiles`, `Gene Counts`
- Singular for reference assets: `Reference Genome`, `Protein DB`

## Node Shape Reference

```mermaid
flowchart TD
    A[FASTQ / file artifact] 
    B([Sample Metadata])
    C[[Reference Database]]
    D(Statistical Object)
    E[[Trained Model]]
    F([Final Report / Figure Set])
```

Syntax:
- `[Text]` rectangle — files, primary data objects
- `([Text])` stadium — metadata, cohort descriptors
- `[[Text]]` double rectangle — reference databases, indices
- `(Text)` rounded — derived matrices, model objects

## Phase Subgraph Template

```mermaid
flowchart TD
    subgraph P1[Preprocessing]
        direction LR
        ...
    end
    subgraph P2[Core Analysis]
        direction LR
        ...
    end
    P1 --> P2
```

Split when total nodes > ~40 or edges become unreadable.

## Multi-Omics Example (abbreviated)

```mermaid
flowchart LR
    FQ[FASTQ] -->|MEGAHIT| MAG[MAG]
    MAG -->|Prodigal| CDS[Predicted CDS]
    CDS -->|DIAMOND vs UniRef| KO[KO Abundance]
    MAG -->|CheckM| QC_MAG[MAG Quality Table]
    KO -->|DESeq2| DE[Differential Expression Table]
```

## Output File Wrapper

Save diagram in markdown:

```markdown
# Workflow Architecture

**Generated:** YYYY-MM-DD
**Source:** Repository scan + method-decision.md
**Coverage:** N verified edges, M inferred edges

## Diagram

​```mermaid
flowchart LR
    ...
​```

## Legend
| Symbol | Meaning |
|--------|---------|
| Solid arrow | Verified in workflow code/config |
| Dashed arrow | Inferred from project structure |
| [[ ]] | Reference database |
| ([ ]) | Metadata |

## Uncertainty Report
[See gaps template below]
```

## Uncertainty Report Template

```markdown
## Workflow reconstruction gaps

### Unconnected artifacts
| Path | Likely type | Issue |
|------|-------------|-------|
| results/beta_pcoa.tsv | PCoA Coordinates | No upstream rule found in Snakefile |

### Inferred links (confirm)
| From | To | Edge label | Basis |
|------|-----|------------|-------|
| Trimmed FASTQ | Taxonomic Profiles | Kraken2 (inferred) | Output dir naming; rule commented out |

### Conflicts
| Source A | Source B | Conflict |
|----------|----------|----------|
| Snakefile | README | README lists MetaPhlAn; code uses Kraken2 |

### Required to complete diagram
- [ ] Confirm host removal step exists or remove node
- [ ] Provide script for `scripts/orphan_step.R`
```

## Readability Guidelines

- Max ~25–40 nodes per diagram; split if larger
- Edge labels: tool name only; params in uncertainty report unless critical
- Avoid crossing edges: reorder nodes or use subgraphs
- Branching: use explicit merge nodes (`Merged Count Matrix`) not implicit joins
- ML flows: show `Feature Matrix -->|XGBoost| Trained Model -->|predict| Predictions`

## Anti-Patterns

| Avoid | Prefer |
|-------|--------|
| Idealized pipeline not in repo | Only verified + strongly inferred edges |
| Raw paths as node labels | Generic object names |
| Invented tool on edge | Unknown → gap report |
| Single giant unreadable graph | Phase subgraphs |
| Hiding inferred edges | Dashed + uncertainty report |
