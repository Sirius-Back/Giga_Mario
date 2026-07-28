# Preprocess reference

Companion to [SKILL.md](SKILL.md). Read when wiring a dataset-specific runner.

## Forbidden

- Do **not** revive `src.pipeline.parse_fasta` / `run_parse_fasta` / `parse_fasta` imports.
- Do **not** call `adapt` with retired `--task` / `--size` — use `--environment` + `--window`.
- Do **not** reimplement adapt / parse_data / parse_target / id_gen / get_mpra / report logic in-chat.
- Do **not** hand-author agentic `parse.md`; only `src.preprocess_report.write_parse_md`.

## Module CLI map

| Module | Typical CLI / API |
|--------|-------------------|
| `src.get_mpra` | `python -m src.get_mpra --tpm <dir> --outfolder <dir> [--mode soft\|continuous] [--n-bins 18] [--per-file-scale] [--scale-01]` |
| `src.pipeline.id_gen` | `python -m src.pipeline.id_gen --gtf <path> --outdir <dir> [--gtf-column gene]` |
| `src.pipeline.id_rule` | `run_id_rule(ids, id_csv, id_col_1=…, id_col_2="ID")` (used inside parse_target / generate_fold) |
| `src.pipeline.adapt` | `python -m src.pipeline.adapt --gtf … --fna … --id-csv … --outdir … --environment gene\|random --window '{"pos1":-100,"pos2":100}'` |
| `src.pipeline.parse_data` | `python -m src.pipeline.parse_data --marked … --outdir … --to-type caduceus\|legnet` |
| `src.pipeline.parse_target` | `python -m src.pipeline.parse_target --target … --id-csv … --outdir … --to-type … [--mappings …]` |
| `src.pipeline.generate_fold` | `python -m src.pipeline.generate_fold --id-csv … --prepare-fold … --outdir …` (when ZSV / prepare_fold given) |
| `src.preprocess_report` | `write_parse_md(outdir, id_csv=…, require_fold=…)` → `{outdir}/parse.md` |

Prefer importing `run_*` / `write_parse_md` from these modules inside `src/run/preprocess_{which_data}.py` rather than shelling out, unless a CLI wrapper is clearer for the dataset.

## Runner skeleton

File: `src/run/preprocess_{which_data}.py`

```python
"""Dataset-specific preprocess orchestrator (write-and-exec by @preprocess)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.get_mpra import run_get_mpra  # if needed
from src.pipeline.id_gen import run_id_gen
from src.pipeline.adapt import run_adapt
from src.pipeline.parse_data import run_parse_data
from src.pipeline.parse_target import run_parse_target
from src.pipeline.generate_fold import run_generate_fold  # if ZSV / prepare_fold
from src.preprocess_report import write_parse_md


def _require_path(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="preprocess orchestrator")
    p.add_argument("--gtf", type=Path, required=True)
    p.add_argument("--fna", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--environment", choices=["gene", "random"], required=True)
    p.add_argument("--window", type=str, required=True, help='JSON e.g. {"pos1":-100,"pos2":100}')
    p.add_argument("--to-type", choices=["caduceus", "legnet"], required=True)
    p.add_argument("--mappings", type=Path, default=None)
    p.add_argument("--prepare-fold", type=Path, default=None)
    p.add_argument("--gtf-column", default="gene")
    # optional get_mpra: --tpm-source, --mpra-out, --n-bins, …
    args = p.parse_args(argv)

    gtf = _require_path(args.gtf, "GTF")
    fna = _require_path(args.fna, "FNA")
    target = _require_path(args.target, "TARGET")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    window = json.loads(args.window)

    id_csv = run_id_gen(gtf, gtf_column=args.gtf_column, outdir=outdir)
    run_adapt(
        gtf, fna, outdir=outdir, id_csv=id_csv,
        environment=args.environment, window=window,
    )
    marked = outdir / "MARKED"
    run_parse_data(marked, outdir=outdir, to_type=args.to_type)
    run_parse_target(
        target, outdir=outdir, id_csv=id_csv, to_type=args.to_type, mappings=args.mappings,
    )
    require_fold = args.prepare_fold is not None
    if require_fold:
        run_generate_fold(id_csv, args.prepare_fold, outdir=outdir)
    write_parse_md(outdir, id_csv=id_csv, require_fold=require_fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
```

Adjust layout (subdirs per stage) to match existing panel conventions under `outdir`; keep `write_parse_md` pointed at the same root it can discover (`ID.csv`, `MARKED/`, `PARSED/`, `PREDICT/`, optional `fold.csv`).

## Result format checks

Before declaring COMPLETED, confirm (directly or via `parse.md` checks):

- `ID.csv` pipe header includes `genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID`
- `MARKED/*.fa` present
- `PARSED/*.ext` present for `to_type`
- `PREDICT/` + `predict.csv` (merged or mapped)
- `fold.csv` when ZSV / `prepare_fold` was requested
- `{outdir}/parse.md` written by `write_parse_md`

## Tests

Novel edits to existing `src.pipeline.*` / `src.get_mpra` / `src.preprocess_report` functions require pytest under `tests/pipeline/` (or adjacent).

```bash
python -m pytest tests/pipeline -q
python -m pytest tests/pipeline/test_preprocess_report.py -q
```
