---
name: go
description: >-
  Metagenome GO differential enrichment: ensure Gene Ontology resources
  (prefer GO.db / local go-basic.obo, fetch if missing), DESeq2 DEG on Bakta GO
  abundances, clusterProfiler enricher on significant terms, volcano and top-20
  LFC±SE barplots. Use for GO enrichment, GO DEG, GO volcano, or Bakta GO DA.
disable-model-invocation: true
---

# GO (metagenome DEG enrichment)

## Purpose

1. **Ensure GO ontology** is available (check pre-downloaded / `GO.db` first; download `go-basic.obo` only if missing)
2. **Metagenome DEG** of Bakta GO-term abundances (**ANCOM-BC2** default; optional DESeq2) + **enricher** over ontology parents
3. **Visualize** DEG as volcano + top-20 |LFC| barplots with SE whiskers

Complementary to `@metabolism` (which **excludes** GO).

Follow: **validation-first**, **reproducibility**, **publication-figures**, **statistical-analysis**, **method-decision-tracking**, **artifact-registry**.

## Related codebase

| Source | Role |
|--------|------|
| Kristina `bakta_gff3.R` / `bakta_function_long.csv` | GO rows (`function_type=go`) |
| Bee `combined_analysis.Rmd` / `GO/code_analysis.R` | `clusterProfiler::enricher`, volcano; OBO download helper |
| Bee / Bioconductor `GO.db` | Preferred term names + ontology graph |
| `ixodes/.../ticks_metaanalyse.Rmd` | LFC ± SE `geom_errorbar` pattern |

## Input resolution

1. `--long` Bakta long CSV/TSV (uses `function_type == go` only)
2. Else `--deg` precomputed DEG table (`go_id`/`gene`, `log2FoldChange`, `lfcSE`, `pvalue`, `padj`) → skip DESeq2, still annotate + plot + enricher
3. Else auto: skill fixture / Kristina `bakta_function_long.csv`

**Metadata** (required unless `--deg`): `--metadata` with `sample` + group column (`--group-col`, default `group`). Phyloseq RDS `--ps-rds` accepted (sample_data → metadata).

## GO resource resolution

1. `GO.db` if installed (preferred for names + ancestor map)
2. Else existing `go-basic.obo` under `--go-cache`, skill `cache/`, `refs/go/`, or CWD `*.obo`
3. Else download `https://purl.obolibrary.org/obo/go/go-basic.obo` into `--go-cache` (unless `--fetch-go false`)

## Defaults

| Parameter | Default |
|-----------|---------|
| Design | `~ group` (**ANCOM-BC2** default; optional `--method deseq2`) |
| Prevalence filter | `--prv-cut 0.1` |
| DEG cutoffs (enricher gene set) | `|log2FC| ≥ 1`, `padj/q < 0.05` |
| Top barplot | 20 by `|log2FoldChange|` |
| Palette | up `#D81B60`, down `#1B9E77`, NS grey |

## Workflow

```
GO:
- [ ] Step 1: Ensure GO.db and/or go-basic.obo (fetch only if missing)
- [ ] Step 2: Resolve GO count matrix + metadata (or --deg)
- [ ] Step 3: ANCOM-BC2 DEG (default); annotate GO names; write deg table
- [ ] Step 4: enricher on significant GO IDs (ontology parents)
- [ ] Step 5: Volcano + top-20 LFC±SE bars (PDF/PNG)
- [ ] Step 6: go-report.json
```

## Executable

```bash
Rscript .cursor/skills/go/scripts/go.R \
  --long PATH/bakta_function_long.csv \
  --metadata PATH/metadata.csv \
  --group-col group \
  --outdir test/go/run

Rscript .cursor/skills/go/scripts/go.R --self-test
```

Optional: `--method ancombc|deseq2` `--deg PATH` `--ps-rds phyloseq.rds` `--prv-cut 0.1` `--lfc-cut 1` `--padj-cut 0.05` `--top-n 20` `--go-cache refs/go` `--fetch-go true` `--contrast Older,Young`

Thin hook: `.cursor/hooks/go.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `go_counts.tsv` | GO × sample integer counts (if computed) |
| `go_deg.tsv` | DEG table with names, LFC, lfcSE, p, padj |
| `go_enrichment.tsv` | clusterProfiler enricher results |
| `go_volcano.pdf` / `.png` | Volcano |
| `go_top20_lfc.pdf` / `.png` | Top-20 LFC ± SE bars |
| `go-report.json` | Resources used, n terms, figure paths |
