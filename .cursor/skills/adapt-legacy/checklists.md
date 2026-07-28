# Adapt — checklists

## Pre-flight

- [ ] Input path exists (or auto-detect succeeds)
- [ ] `window_size > 2 * flank`
- [ ] Env has Python + pyfaidx (or equivalent)
- [ ] `docs/caduceus_format.md` present or will be written/refreshed

## Post-run

- [ ] `adapt/samples.tsv` non-empty
- [ ] `adapt/labels.tsv` row count == accepted samples
- [ ] `adapt/excluded_genes.tsv` present
- [ ] `adapt/statistics.json` accepted+rejected match tables
- [ ] `adapt/qc_report.md` written
- [ ] `adapt/caduceus_ready/` consumable by `/caduceus`
- [ ] Artifacts registered; `method-decision.md` updated
