# Adapt — README

Mandatory **Caduceus-prep** stage: raw or split genomes → gene±200 bp DNA windows + continuous TPM.

Does **not** create train/val/test folds (`@split` owns that).

## Quick start

```bash
conda run -n caduceus_env python .cursor/skills/adapt/scripts/adapt.py \
  --input auto \
  --out adapt \
  --window-size 8192
```

Auto-detects:

- Split panel: `data_splits/full/{train,val,test}/<genome>/` with `genome.fna`, `genes.tsv`, `expression_tpm.csv`
- Raw / reformat manifests when no splits are present

## Outputs (`adapt/`)

| File | Role |
|------|------|
| `manifest.tsv` | Genome / fold inventory |
| `samples.tsv` | One row per accepted gene window (+ sequence) |
| `labels.tsv` | `sample_id`, `TPM` (continuous) |
| `excluded_genes.tsv` | Rejected genes + reason |
| `metadata.json` | Run provenance |
| `config.yaml` | Frozen run config |
| `statistics.json` | Accepted/rejected rates per genome |
| `qc_report.md` | QC findings |
| `README.md` | Dataset README |
| `METHOD_DECISIONS.md` | Local method lock copy |
| `caduceus_ready/` | Sequences + labels for `/caduceus` |

## Related docs

- Project: `docs/adapt.md`, `docs/caduceus_format.md`
- Skill: [SKILL.md](SKILL.md), [examples.md](examples.md), [workflow.md](workflow.md)
