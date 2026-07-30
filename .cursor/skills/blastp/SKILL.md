---
name: blastp
description: >-
  Define and implement the BLASTP homology split (splits/blastp.md): adapt from
  fna+gtf (not panel MARKED), filter IDs to PARSED, then SBS fold-filter →
  blastp (optional non-all-vs-all heuristic) → cluster/stratify. Use when the
  user mentions blastp, BLASTP split, protein homology folds, genetic-code
  translation for splits, or asks to /split-generate type=blastp.
disable-model-invocation: true
---

# BLASTP split (`splits/blastp.md`)

## Purpose

Caption + implementation contract for **protein-homology** fold assignment.
`/split-generate` reads [`splits/blastp.md`](../../../splits/blastp.md) and
writes `src/splits/blastp.py` for `/split` (`split-predict type=blastp`).

Follow: **validation-first**, **reproducibility**, **missing-data-policy**,
**scientific-integrity**, **artifact-registry**, **method-decision-tracking**,
**skills-write-and-exec-src**. Prefer **local-job-queue** for large BLASTP jobs.

Do **not** reimplement SBS assign/strat/materialize in chat. Reuse
`src.splits.sbs` + `src.pipeline.adapt` / `split_predict` / `split`.

## Obligatory caption

**Always** treat `splits/blastp.md` as the user-facing and generate-facing
spec. If missing or incomplete, stop and ask — do not invent roles or ratios.

## Inputs (locked by caption)

| Input | Role |
|-------|------|
| **fna**, **gtf** | Raw genomes — **instead of** panel `MARKED/` |
| **window** | Adapt window |
| **genetic code** | DNA→protein; default **`universal`** |
| **PARSED** | Filter keep-set (IDs that exist in PARSED) |
| **fold.csv** | Optional ZSV / fold filter |
| **stratification.csv** | Optional; SBS fold-grain strat |

## Processing (must match caption)

```
(1) adapt (raw -> MARKED + intersect.py)
(2) filter (надо написать): берем только те ID которые есть в PARSED
(3) sbs:
(3.1) фильтруем фолд
(3.2) делаем blastp: можем сразу тут реализовать эвристику, чтобы не делать все-со-всеми
(3.3) делаем кластеризацию, контроль, стратифицируем (там уже in-built есть)
```

### Step map → code

| Step | Action | Reuse |
|------|--------|-------|
| **(1) adapt** | `raw` → `MARKED` + `intersect.csv` | `src.pipeline.adapt` / `@preprocess` |
| **(2) filter** | Keep IDs ∈ `PARSED` only | **Write** thin helper (mirror `intersect_pangenome` / `src.splits.pangenome.intersect_pangenome`); do not invent IDs |
| **(3.1) fold filter** | Hold out ZSV; restrict clustering set | SBS / `fold.csv` contracts |
| **(3.2) blastp** | Protein search; optional **heuristic** to avoid all-vs-all | BLAST+ (`blastp`, `makeblastdb`); feature/edge table → SBS, not dense \(n\times n\) clustering requirement |
| **(3.3) cluster + control + strat** | Folds → train/val/test | **In-built SBS** (`assign_from_features`, strat aggregate, PCA diagnostics) |

## /split-generate checklist

```
blastp split-generate:
- [ ] splits/blastp.md present (this skill + caption)
- [ ] method-decision: genetic code default=universal; BLASTP heuristic choice
- [ ] WRITE filter helper (PARSED ∩ MARKED)
- [ ] WRITE src/splits/blastp.py + wire type=blastp
- [ ] Reuse SBS assign/viz; do not fork materialize
- [ ] pytest for novel filter / blastp feature path
- [ ] artifact-registry + DONE for /split
```

## Rules

1. Inputs are **fna + gtf + window + genetic code** — not silent panel MARKED reuse.
2. Filter step is **required new code** until implemented; IDs must exist in PARSED.
3. Prefer non-all-vs-all BLASTP heuristic when scaling; document choice in method-decision.
4. Clustering / QC / stratification: **SBS in-built only**.
5. Novel `./src` → pytest before COMPLETED.
6. Large BLASTP → `queue.md` (`cpu_ram_heavy` or GPU-irrelevant heavy CPU).

## Coordination

| Path / skill | Role |
|--------------|------|
| `splits/blastp.md` | Spec caption (users + `/split-generate`) |
| `/split-generate` | Emits `src/splits/blastp.py` from caption |
| `/split` | split-predict + materialize `SPLIT/` |
| `@preprocess` / `adapt` | Step (1) raw → MARKED + intersect |
| `wiki/sbs.md` | SBS C1/C2 contracts |
| Peer | `splits/pangenome.md` (adapt+filter pattern), `splits/hashfrag.md` (alignment homology) |

## Additional resources

- Caption: [`splits/blastp.md`](../../../splits/blastp.md)
- Split generate: [`../split-generate/SKILL.md`](../split-generate/SKILL.md)
- Preprocess: [`../preprocess/SKILL.md`](../preprocess/SKILL.md)
- SBS: [`wiki/sbs.md`](../../../wiki/sbs.md)
