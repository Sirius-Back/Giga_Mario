---
name: difftree-ggdiffclade
description: >-
  Differential cladogram + abundance boxes via MicrobiotaProcess
  (diff_analysis, ggdiffclade, ggdiffbox) — PacBio two-panel layout with
  ggdiffclade legend at bottom. Use when the user mentions
  difftree-ggdiffclade, ggdiffclade, ggdiffbox, MicrobiotaProcess LDA cladogram,
  or PacBio-style DE tree.
disable-model-invocation: true
---

# Diff Tree (ggdiffclade)

## Purpose

Reproduce the **PacBio** `Article.Rmd` DE figure: `diff_analysis` → `ggdiffclade` (radial) + `ggdiffbox`, combined with `ggarrange(widths = c(0.7, 1))`.

Unlike `@difftree-ggtree` (ANCOM-BC + ggtree), this skill **runs** MicrobiotaProcess LDA differential analysis and plots with **MicrobiotaProcess** geoms.

Follow: **validation-first**, **reproducibility**, **publication-figures**, **method-decision-tracking**, **slurm-execution-policy**, **artifact-registry**.

## Source code (exact)

| Role | Path | Lines |
|------|------|-------|
| **Primary** | `/mnt/tank/scratch/dsmutin/ixodes/PacBio/Article.Rmd` | genus DE ~1876–1970; sex DE ~2099–2160+ |
| Copy | `/mnt/tank/scratch/dsmutin/archive/bioinformatics/2026/Kristina/codebase.Rmd` | genus ~1709–1803; sex ~1932–2026 |
| Palette | PacBio `Article.Rmd` | `colors_2 <- c("lightgreen","pink")` ~L185 |

## Layout (Locked)

Two-panel plot with **ggdiffclade legend at bottom**:

1. **Left** — `ggdiffclade(..., layout = "radial")` (no local legend)
2. **Right** — `ggdiffbox` abundance + LDA (`legend.position = "none"` on both; LDA drops shared y-axis text)
3. **Bottom** — fill / size legends from ggdiffclade only

Taxon labels are trimmed: long `s__uncultured…_o__Name` chains → `s - uncultured Name ASVk`; all-uncultured → `s - unknown ASVk`; numeric/code tips (e.g. `1921-2`, `AD3`) keep plain (non-italic) labels on both tree and boxplot.

```r
ggarrange(
  ggarrange(diffclade, diffbox, widths = c(0.7, 1), legend = "none"),
  common_legend,
  ncol = 1, heights = c(1, 0.14)
)
```

## Defaults (PacBio-aligned)

| Parameter | Default |
|-----------|---------|
| `mlfun` | `lda` |
| First test | `kruskal_test` (`coin`), `firstalpha = 0.05` |
| Second test | `wilcox_test`, `secondalpha = 0.01`, `subclwilc = TRUE` |
| `subclmin` | `3` |
| `strictmod` | `TRUE` |
| `lda` | `3` |
| Clade | `layout = "radial"`, `taxlevel = 3`, `removeUnknown = TRUE`, `reduce = TRUE` |

## Input

1. `--rds` rarefied phyloseq (preferred)
2. Else `grazing_phyloseq_rare.rds` / rarefaction-analysis rare RDS
3. `--target` or RDS `$target` (2+ levels; coerced to factor)

**Preprocess for MicrobiotaProcess:** drop `phy_tree` and non-standard ranks (`tip_rank`); strip SILVA `^[a-z]__` prefixes. (Keeps MP taxonomy-tree build stable under current ggtree/ape.)

## Workflow

```
difftree-ggdiffclade:
- [ ] Step 1: Resolve rarefied phyloseq + target
- [ ] Step 2: prepare_ps_for_mp (ranks, factor target, no phy_tree)
- [ ] Step 3: MicrobiotaProcess::diff_analysis (coin tests)
- [ ] Step 4: ggdiffclade + ggdiffbox; legend on right; ggarrange 0.7∶1
- [ ] Step 5: Write PDF/PNG + result TSV + report JSON; register artifacts
```

## Executable

```bash
Rscript .cursor/skills/difftree-ggdiffclade/scripts/difftree_ggdiffclade.R \
  --outdir test/difftree-ggdiffclade/grazing \
  --target grazing

Rscript .cursor/skills/difftree-ggdiffclade/scripts/difftree_ggdiffclade.R --self-test

sbatch .cursor/skills/difftree-ggdiffclade/scripts/difftree_ggdiffclade.sbatch
```

Optional: `--rds` `--taxlevel 3` `--firstalpha 0.05` `--secondalpha 0.01` `--subclmin 3` `--lda 3` `--seed 42`

Thin hook: `.cursor/hooks/difftree-ggdiffclade.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `difftree_ggdiffclade_<target>.pdf/.png` | Two-panel combined figure |
| `diff_analysis_result.tsv` | `deres@result` |
| `difftree-ggdiffclade-report.json` | Params, versions, paths |

## Related

- `@difftree-ggtree` — ANCOM-BC cladobox analogue (does **not** re-run DA)
- `@difftree-metacoder` — heat trees
- See [reference.md](reference.md) for PacBio code excerpts
