---
name: analyze-ready-data
description: >-
  EDA for ready-panel windows: barplots (non-coding / normal coding / large coding)
  and GC + length density plots via src/ready_analysis.py. Use for /analyze-ready-data,
  ready/ QC, or any data_ready-like directory with non_coding.csv + large_genes.csv.
disable-model-invocation: true
---

# Analyze Ready Data

## Purpose

Classify ready-panel DNA windows into **non-coding**, **normal coding**, and **large coding**,
then write publication-style **barplots** (counts) and **density plots** (GC; length).

Follow: validation-first, missing-data-policy, reproducibility, publication-figures,
artifact-registry, method-decision-tracking, scientific-integrity.

## Code-first contract (LOCKED)

```
@analyze-ready-data cycle:
  1. Resolve READY_DIR (ready/, data_ready/, or another panel with the same CSV schema)
  2. WRITE / UPDATE src/ready_analysis.py only if behavior must change
  3. EXEC: python -m src.ready_analysis --ready-dir <READY_DIR> --outdir <OUT>
  4. REUSE the same module for other ready-like panels (change --ready-dir / --outdir)
```

**Never** reimplement grouping or plots in-chat. **Never** invent counts or densities —
only report what the script wrote under `--outdir`.

Module path: `src/ready_analysis.py` → `python -m src.ready_analysis`.

## Required inputs

| File | Role |
|------|------|
| `<READY_DIR>/non_coding.csv` | Pipe-separated; columns `GeneOrID\|Chr\|Position_start\|Position_end\|Length\|GC\|kind\|Genome`. `kind` ∈ `{gene, non_coding}` |
| `<READY_DIR>/large_genes.csv` | Pipe-separated large CDS crops; join key `Genome+Gene+Chr+Window_start+Window_end` ↔ gene windows |

If either file is missing, empty, or schema-invalid → **stop** (script fails early). Do not guess groups.

Default `READY_DIR`: `ready` (often symlink → `data_ready`).

## Group definitions (locked)

| Group | Rule |
|-------|------|
| **non-coding** | `kind == non_coding` |
| **large coding** | `kind == gene` and exact window match to `large_genes.csv` (CDS > 130 kb crops) |
| **normal coding** | `kind == gene` and not large |

## Output

Under `--outdir` (default `output/ready_analysis/`):

| Artifact | Role |
|----------|------|
| `barplot_group_counts.{pdf,png,svg}` | Overall counts |
| `barplot_group_counts_by_genome.{pdf,png,svg}` | Counts by genome × group |
| `density_gc_by_group.{pdf,png,svg}` | GC fraction KDE overlay |
| `density_length_by_group.{pdf,png,svg}` | Window length KDE overlay |
| `group_counts.csv` | Overall counts |
| `group_counts_by_genome.csv` | Per-genome counts |
| `summary.json` | Counts, means, software versions, group definitions |

Style: Okabe–Ito + distinct linestyles; PDF/SVG + PNG@300 dpi (module `DPI`).

## Exact command

```bash
# Default ready/ panel
conda run -n caduceus_env python -m src.ready_analysis \
  --ready-dir ready \
  --outdir output/ready_analysis

# Another ready-like panel (smoke, alternate prep, etc.)
conda run -n caduceus_env python -m src.ready_analysis \
  --ready-dir data_ready_smoke2 \
  --outdir output/ready_analysis_smoke2
```

| Flag | Default | Notes |
|------|---------|-------|
| `--ready-dir` | `ready` | Any directory with the two CSVs above |
| `--outdir` | `output/ready_analysis` | Create if missing; isolate per dataset |

Seed in module: **42** (reserved for future sampling; keep fixed).

## Workflow checklist

```
analyze-ready-data:
- [ ] Confirm non_coding.csv + large_genes.csv under READY_DIR
- [ ] Update src/ready_analysis.py only if schema/plots must change
- [ ] Exec python -m src.ready_analysis --ready-dir … --outdir …
- [ ] Verify barplots + both density plots + summary.json in outdir
- [ ] Register new/updated artifacts in docs/artifact-registry.md
- [ ] Append method-decision.md if group rules or plot choices changed
```

## Done criteria

- Exit code 0
- All listed artifacts present under `--outdir`
- `summary.json` `counts` match `group_counts.csv`
- Large-gene join: matched windows == `len(large_genes.csv)` (enforced in module)

## Anti-patterns

- Replotting with ad-hoc one-off scripts instead of `src/ready_analysis.py`
- Hard-coding absolute machine paths in the module
- Mixing two panels into one `--outdir` without a clear supersede note
- Inferring large coding from Length alone (must join `large_genes.csv`)
