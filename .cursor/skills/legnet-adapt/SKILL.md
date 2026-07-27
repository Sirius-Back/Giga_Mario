---
name: legnet-adapt
description: >-
  Prepare human_legnet-ready 230 bp promoters from raw FNA/GTF/TPM or BED+FASTA:
  TSS-centered 200 bp CRS, lentiMPRA adapter stitch, continuous TPM; write
  legnet_ready/ via src/legnet_preprocess.py. Never assigns project train/val/test folds.
disable-model-invocation: true
---

# LegNet Adapt

## Goal

`legnet-adapt` prepares **human_legnet**-ready DNA sequences from genomic data.

```
raw/{fna,gtf,tpm}  ──►  src/legnet_preprocess.py  ──►  legnet_ready/
                                                         ├── promoters.bed
                                                         ├── all.tsv / {GCF}.tsv
                                                         ├── sequences.fa
                                                         └── statistics.json / metadata.json
```

Or parse an existing BED:

```
promoters.bed + genome.fna  ──►  same outputs (adapter stitch to 230 bp)
```

`@split` owns Caduceus project folds. This skill MUST NOT perform project train/validation/test splitting. The TSV `fold` column is `(hash%10)+1` → **1..10** for human_legnet CV compatibility only.

--------------------------------------------------
GENERAL PRINCIPLES
--------------------------------------------------

Reproducible, deterministic (seed 42), restartable, idempotent, scientific, fully documented.

Never silently guess. Follow: validation-first, missing-data-policy, reproducibility, scientific-integrity, method-decision-tracking, artifact-registry, task-status.

--------------------------------------------------
STAGE 1 — REPOSITORY AUDIT
--------------------------------------------------

Expect:

```
raw/fna/*.fna(.gz)
raw/gtf/*.gtf(.gz)
raw/tpm/*.csv
raw/random_borzoi_expr_file_mappings.csv
```

Pair by GCF accession (reuse `@adapt` discovery). Skip genomes missing a local TPM. Abort if no complete bundles.

Vendor installs (required for training later; prep only needs pyfaidx):

- `software/human_legnet` — primary (230 bp)
- `software/LegNet` — upstream reference
- conda env `legnet` from `software/human_legnet/envs/environment.yml`

--------------------------------------------------
STAGE 2 — PROMOTER + STITCH (LOCKED 2026-07-27)
--------------------------------------------------

| Rule | Value |
|------|-------|
| Anchor | Strand-aware **TSS** = GTF `gene` start (`+`) or end (`-`) |
| CRS | **200 bp** centered on TSS: `[TSS−100, TSS+100)` 0-based; skip if incomplete |
| Orientation | Gene-oriented sequence (RC if strand `-`) |
| Stitch | `AGGACCGGATCAACT` + CRS + `CATTGCGTGAACCGA` → **230 bp** |
| Label | Continuous **TPM** (never invent) |
| BED | `chrom start end name score strand` (score=TPM) |
| TSV | `seq_id seq mean_value fold rev` (human_legnet) |
| fold | `(stable_hash(seed:seq) % 10) + 1` → **1..10** — not `@split` |

--------------------------------------------------
OUTPUT
--------------------------------------------------

Directory: **`legnet_ready/`** (see [wiki/legnet_conversion.md](../../wiki/legnet_conversion.md))

--------------------------------------------------
EXACT COMMAND
--------------------------------------------------

**Do not reimplement windowing/stitching in-chat** — run the script.

```bash
conda run -n legnet python src/legnet_preprocess.py \
  --raw raw \
  --out legnet_ready \
  --crs-bp 200 \
  --seed 42
```

Parse existing BED:

```bash
conda run -n legnet python src/legnet_preprocess.py \
  --bed promoters.bed \
  --fasta path/to/genome.fna \
  --out legnet_ready \
  --stitch-adapters
```

| Flag | Default | Notes |
|------|---------|-------|
| `--raw` | `raw` | Root with `fna/`, `gtf/`, `tpm/` |
| `--out` | `legnet_ready` | Output directory |
| `--crs-bp` | `200` | Locked for human_legnet |
| `--seed` | `42` | Fold hash seed |
| `--bed` / `--fasta` | off | BED parse mode |
| `--stitch-adapters` | on | Wrap to 230 bp |
| `--genomes` | all | Optional GCF filter |
| `--max-genes` | none | Smoke-test cap |

If env `legnet` is unavailable, `caduceus_env` with pyfaidx also runs the prep script.

## Workflow checklist

```
legnet-adapt:
- [ ] Stage 1: Audit raw/{fna,gtf,tpm} + mapping; skip incomplete; abort if empty
- [ ] Confirm software/human_legnet + software/LegNet present
- [ ] Stage 2: TSS±100 CRS; gene orientation; adapter stitch → 230 bp
- [ ] Write legnet_ready/{promoters.bed,all.tsv,sequences.fa,statistics.json}
- [ ] Update method-decision.md + artifact-registry + wiki/legnet_conversion.md
```

## Additional resources

- [wiki/legnet_conversion.md](../../wiki/legnet_conversion.md)
- [README.md](README.md)
- Project entry: [`src/legnet_preprocess.py`](../../src/legnet_preprocess.py)
- Next: [`@legnet`](../legnet/SKILL.md) / [`src/legnet.py`](../../src/legnet.py) — train on `legnet_ready/`

