---
name: preprocess
description: >-
  Combined genomic preprocess: fail-early input checks, then write-and-exec
  src/run/preprocess_{which_data}.py calling get_mpra, id_gen/id_rule, adapt
  (environment/window), parse_data, parse_target, optional generate_fold (ZSV),
  format checks, and src.preprocess_report.write_parse_md → parse.md. Use for
  /preprocess. Combines prior adapt / prepare-target / legnet-adapt / get_mpra.
disable-model-invocation: true
---

# Preprocess

## Goal

One `@preprocess` skill that replaces the prior split of **adapt / prepare-target / legnet-adapt / get_mpra** into a single write-and-exec runner under `src/run/`.

```
GTF + FNA + TARGET (+ optional MPRA / mappings / ZSV)
  → src/run/preprocess_{which_data}.py
  → ID.csv, MARKED/, PARSED/, PREDICT/, optional fold.csv, parse.md
```

Does **not** run split-predict / split / train. Downstream: `@split` or pipeline split stages.

## Rules (must follow)

- Skills reuse `./src` code; do not reimplement stages in-chat.
- If editing `./src`, keep same general functions so legacy does not break; do not invent parallel APIs when existing ones cover the role.
- Write production-ready code under `./src` / `./src/run/`.
- Novel edits to existing functions require pytest.

## Obligatory inputs

Document / require these before writing or executing the runner. **Stop** (validation-first / missing-data-policy) if any required path is missing, empty, or unreadable — do not guess.

| Input | Required | Role |
|-------|----------|------|
| `which_data` | **yes** | Dataset name → runner path `src/run/preprocess_{which_data}.py` |
| GTF path/folder | **yes** | Annotation(s) for `id_gen` + `adapt` |
| FNA path/folder | **yes** | Genome FASTA(s) for `adapt` |
| TARGET folder | **yes** | TPM/MPRA wide CSVs for `parse_target` (or post-`get_mpra` folder) |
| `outdir` | **yes** | Output root for stages + `parse.md` |

### Optional inputs

| Input | Role |
|-------|------|
| `--mappings` | Sample mapping CSV for mapped PREDICT (`parse_target`) |
| ZSV / `prepare_fold` | If ZSV holdouts are specified → run `generate_fold` (`ID.csv` + `prepare_fold.csv` → `fold.csv`) |
| `environment` / `window` | `adapt` API: `environment=gene\|random`; `window={"pos1":…,"pos2":…}` — **not** retired `task` |
| `to_type` | `caduceus` \| `legnet` for `parse_data` / `parse_target` |
| `get_mpra` flags | When LegNet targets are needed: `--tpm`, `--outfolder`, `--mode soft|continuous`, `--n-bins`, `--per-file-scale`, `--scale-01` via `src.get_mpra` |

Ask the user for any missing obligatory input. Do not invent defaults for GTF/FNA/TARGET/`which_data`/`outdir`.

## Architecture (write then exec)

```
@preprocess:
  1. Check input data (fail early if missing)
  2. WRITE EXACT code → ./src/run/preprocess_{which_data}.py
  3. EXEC that runner
  4. Ensure {outdir}/parse.md exists (from src.preprocess_report.write_parse_md)
```

### Runner must call existing modules

Thin imports only — no reimplemented windowing, joins, or reports in-chat or inside the runner beyond orchestration:

| Stage | Module | Notes |
|-------|--------|-------|
| MPRA (if needed) | `src.get_mpra` | TPM → bin-fraction TARGET folder |
| IDs | `src.pipeline.id_gen`, `src.pipeline.id_rule` | `ID.csv`; remaps via `id_rule` |
| Mark | `src.pipeline.adapt` | Use **`environment` / `window`** API — not retired `task` |
| Sequences | `src.pipeline.parse_data` | Canonical; **`parse_fasta` was removed — do NOT revive `parse_fasta`** |
| Labels | `src.pipeline.parse_target` | TARGET → PREDICT |
| Folds | `src.pipeline.generate_fold` | Only if ZSV / `prepare_fold` is specified |
| Report | `src.preprocess_report.write_parse_md` | Plain code only — not agentic prose generation |

Also: **check result formats** (ID.csv columns, MARKED FASTA, PARSED/PREDICT layouts, optional `fold.csv`) before/via the report writer.

### After exec

During real skilled work: after exec, ensure `parse.md` exists from that reporter (`write_parse_md`). If missing → re-call `write_parse_md` or fail; do not hand-write agentic `parse.md`.

## Exact exec pattern

```bash
# After writing src/run/preprocess_{which_data}.py
python -m src.run.preprocess_{which_data} \
  --gtf path/to/gtf \
  --fna path/to/fna \
  --target path/to/tpm_or_mpra \
  --outdir path/to/out \
  --environment gene \
  --window '{"pos1":-100,"pos2":100}' \
  --to-type caduceus
# optional: --mappings … --prepare-fold … --get-mpra / get_mpra flags
```

Prefer `conda run -n <env>` when the project env is required (pyfaidx, etc.).

Smoke / contracts:

```bash
python -m pytest tests/pipeline -q
python -m src.preprocess_report --outdir path/to/out
```

## Workflow checklist

```
preprocess:
- [ ] Collect obligatory inputs (which_data, GTF, FNA, TARGET, outdir)
- [ ] Fail early if any required input missing/empty
- [ ] WRITE src/run/preprocess_{which_data}.py (imports existing modules only)
- [ ] EXEC the runner
- [ ] Confirm MARKED / PARSED / PREDICT (+ optional fold.csv)
- [ ] Confirm {outdir}/parse.md from write_parse_md
- [ ] Novel src edits → pytest; method-decision + artifact-registry as needed
```

## Coordination

| Skill / module | Role |
|----------------|------|
| `@adapt` / `src.pipeline.adapt` | MARKED + intersect (environment/window) |
| `@prepare-target` / `src.pipeline.parse_target` | TARGET → PREDICT |
| `@legnet-adapt` | Legacy LegNet-only path; prefer `@preprocess` + `to_type=legnet` |
| `src.get_mpra` | Optional TPM → MPRA bin fractions |
| `@adapt-legacy` / `src.preprocessing` | Old `data_ready/` Caduceus panel — not this skill |
| `wiki/architecture.md` | Stage contracts |

## Additional resources

- [reference.md](reference.md) — runner skeleton, CLI map, forbidden APIs
- Contracts: [`wiki/architecture.md`](../../wiki/architecture.md)
- Report: [`src/preprocess_report.py`](../../src/preprocess_report.py)
