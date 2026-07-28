---
name: genome-fna-gtf-reformat
description: >-
  Index paired genome .fna/.gtf files, optionally subsample distinct species,
  and reformat into sample manifests and fold-ready layouts for @split /
  Caduceus-style random splits. Use when adapting genomes_smoketest-style
  FASTA+GTF directories for train/val/test splits or GenomicBenchmarks-compatible
  sample tables.
disable-model-invocation: true
---

# Genome FNA/GTF reformat

## Purpose

Adapt directories of paired prokaryotic/eukaryotic genome assemblies
(`{sample}.fna` + `{sample}.gtf`) into structures usable by `@split` and
Caduceus / GenomicBenchmarks-style random splits.

Does **not** invent split assignments — only prepares inputs. Fold assignment
belongs to `@split` + `splits/*.md` (+ implementation `run`).

Follow: **validation-first**, **missing-data-policy**, **reproducibility**,
**method-decision-tracking**, **artifact-registry**, **task-status**.

## When to use

- Input is a flat or nested dir of `*.fna` / `*.gtf` pairs (e.g. `genomes_smoketest/`)
- Downstream needs a sample table, species labels, or fold directories
- Caduceus GenomicBenchmark loaders expect train/test trees of sequence files,
  not raw NCBI assemblies

## Required inputs

| Input | Meaning |
|-------|---------|
| **SRC** | Directory containing paired `.fna` and `.gtf` files |
| **OUT** | Project-relative output root for manifests / intermediate layout |

Optional: `N` genomes, `distinct_species=true`, `seed`, `label_col` (default species).

## Pair discovery

1. List `*.fna` under `SRC` (non-recursive unless user asks).
2. For each stem `S`, require `S.gtf` beside `S.fna`. Fail early if unpaired.
3. Parse **species** from filename when possible:
   - Prefer longest prefix before the last `_ASM…` / `_NCTC…` / assembly token
   - Else treat stem as `sample_id` and set `species=unknown` (do not guess taxonomy online unless user requested public search)
4. Validate non-empty FNA (FASTA header `>`) and non-empty GTF.

## Outputs (write under `OUT`)

| Path | Content |
|------|---------|
| `manifest.tsv` | Columns: `sample_id`, `species`, `fna_path`, `gtf_path` (relative to project root) |
| `selected.tsv` | Subset after optional N / distinct-species filter; include `seed` in sidecar |
| `selection.json` | `{seed, n, distinct_species, selected_ids[]}` |
| Optional `gb_layout/{train,test}/…` | Only if a Caduceus/GB dry-run needs sequence files — one plain-seq file per sample under a single label class (e.g. `genome/`), **without** inventing train/test membership |

Do **not** write final `data_splits/.../{train,val,test}` here — that is `@split`.

## Distinct-species subsample

When `N` genomes and `distinct_species=true`:

1. Group by `species`.
2. If fewer unique species than `N` → **stop** (missing-data-policy); report counts.
3. With fixed `seed` (`numpy.random.default_rng(seed)` or `random.Random(seed)`), sample one genome per species, then if `N` < n_species sample `N` species without replacement.
4. Record choice in `method-decision.md`.

## Caduceus / GenomicBenchmarks notes

Upstream Caduceus `GenomicBenchmarkDataset` expects:

```
{dest}/{dataset_name}/{split}/{label}/*.txt   # raw DNA sequence per file
```

and creates **val** via `torch.utils.data.random_split` on train (seed =
`dataset.train_val_split_seed`). Whole assemblies are **samples**, not intervals:
copy or symlink `.fna`/`.gtf` into fold dirs after `@split` assigns membership;
do not truncate genomes unless the user asks.

For N=3 with train+val+test required, prefer **one sample per fold** after a
seeded shuffle (document in `method-decision.md`) rather than forcing 90/10 on
tiny N.

## Workflow

```
genome-fna-gtf-reformat:
- [ ] Validate SRC exists; list pairs; fail on orphans/empty
- [ ] Write manifest.tsv
- [ ] Apply N / distinct_species selection with fixed seed → selected.tsv + selection.json
- [ ] Record method-decision.md entry
- [ ] Register manifests in artifact-registry.md
- [ ] Hand off selected paths to @split (do not assign folds here)
```

## Method decision (required when choosing)

Append to `method-decision.md`: species parsing rule, seed, N, whether
distinct-species was applied, and why GB layout was or was not materialized.

## Coordination

| Skill | Role |
|-------|------|
| `@split` | Owns fold assignment + `data_splits/...` writes |
| `@dataset-auditor` | Optional QC after manifest |
| `@caduceus` | Consumes folds later; does not replace this reformat |
| `@do-fast` | Runs multi-step pipelines that include this skill |
