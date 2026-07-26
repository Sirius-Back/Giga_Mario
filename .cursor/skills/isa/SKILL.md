---
name: isa
description: >-
  Indicator Species Analysis (indicspecies multipatt) with grazing Figure 3
  panel set: three Euler diagrams, top-N indicator bars (p / IndVal / abundance),
  t-SNE pie markers, sPLS-DA scores + loadings, combined via ggarrange. Use when
  the user mentions ISA, indicator species, multipatt, or grazing-style Fig. 3.
disable-model-invocation: true
---

# ISA (Indicator Species Analysis)

## Purpose

Reproduce the grazing `Article.Rmd` **Figure 3** analysis and layout:

| Panel | Content |
|-------|---------|
| (a) | Euler: all taxa presence across target levels |
| (b) | Euler: significant ISA indicator membership |
| (c) | Euler: occurrence of indicator taxa across levels |
| (d) | Top-N indicators: p-value bars \| IndVal bars \| abundance boxplots |
| (e) | t-SNE of taxa with ISA pie fills |
| (f) | sPLS-DA sample scores |
| (g) | sPLS-DA loadings (ISA-colored) |

Combine **exactly** as grazing:

```r
ggarrange(
  ggarrange(fig3a, fig3b, fig3c, labels = c("(a)", "(b)", "(c)"), nrow = 1, align = "h"),
  ggarrange(fig3d2, fig3d1, fig3d3, labels = c("(d)", "", ""),
            widths = c(1, 5, 5), legend = "none", align = "h", nrow = 1),
  ggarrange(fig3e, fig3f, fig3g, widths = c(1, 1, 2), nrow = 1,
            labels = c("(e)", "(f)", "(g)"), legend = "none"),
  ggarrange(get_legend(fig3d2), get_legend(fig3d1), align = "h"),
  align = "hv", nrow = 4, heights = c(1, 2.2, 1, 0.2)
)
```

Follow: **validation-first**, **reproducibility**, **statistical-analysis**, **publication-figures**, **method-decision-tracking**, **slurm-execution-policy**, **artifact-registry**.

## Input resolution

1. `--rds` if given (prefer **rarefied** counts)
2. Else `*_rare.rds` / `grazing_phyloseq_rare.rds` / `test/rarefaction-analysis/**/phyloseq_rare_*.rds`
3. Else raw counts (will rarefy inside with `--depth` or min sample sum)

Reject MMUPHin relative (`*_batchadj.rds`) unless `--allow-relative true`.

## Defaults (grazing-aligned)

| Parameter | Default |
|-----------|---------|
| `multipatt` `func` | `r.g` |
| Permutations | `9999` (`permute::how`) |
| Significance | `p.value < 0.05` |
| Top-N for (d) | `30` (grazing code; caption said 40) |
| sPLS-DA | `ncomp=2`, `keepX=c(10,10)` on rarefied counts |
| t-SNE | `perplexity=15`, `theta=0.95` |
| Target | RDS `$target` or `--target` / `--targets` (first) |

Target must have **2–3 levels** for the three Euler panels (grazing uses 3). Short labels `S1`…`Sn` are derived for Euler/fill (grazing `Condition2`).

## Workflow

```
ISA:
- [ ] Step 1: Resolve rarefied phyloseq + target (drop NA/empty targets)
- [ ] Step 2: Tip labels (`finalize_taxonomy` / `taxon_format`)
- [ ] Step 3: Relative-abundance multipatt; write ISA table + p-value audit
- [ ] Step 4: Build fig3a–g (−log10(p) for panel d)
- [ ] Step 5: ggarrange combined figure (grazing layout)
- [ ] Step 6: Write isa-report.json; register artifacts
```

## Executable

```bash
Rscript .cursor/skills/isa/scripts/isa.R \
  --outdir test/isa/grazing \
  --target grazing

Rscript .cursor/skills/isa/scripts/isa.R --self-test

sbatch .cursor/skills/isa/scripts/isa.sbatch
```

Optional: `--rds PATH` `--top-n 30` `--nperm 9999` `--depth 1193` `--seed 123` `--keepX 10,10`

Thin hook: `.cursor/hooks/isa.sh`

## Outputs

| Artifact | Content |
|----------|---------|
| `ISA_sp.csv` | Significant indicators + taxonomy (grazing name) |
| `isa_long.tsv` | Long membership table (no NA Condition/Condition2) |
| `isa_pvalue_audit.tsv` | Top-N OTU `p.value`, `−log10(p)`, permutation-floor flags |
| `isa_figure3.pdf` / `.png` | Combined grazing layout |
| `isa_fig3{a-g}_*` | Individual panels |
| `isa-report.json` | Params, `n_unique_p_plotted`, `n_at_perm_floor`, paths |

Samples with NA/empty target are dropped before Euler/multipatt. Panel (d) p-bars use **−log10(p)**. Production `--nperm 9999` (self-test may use 99 → many p at floor `1/(nperm+1)`).

| `isa-report.json` | Paths, n_sig, target levels, params |

## Additional resources

- Panel construction notes: [reference.md](reference.md)
