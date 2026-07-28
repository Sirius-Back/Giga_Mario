"""Generate stratification.csv from ID.csv + prepare_strat.csv (stub).

When implemented, resolve identificators via ``id_rule`` (``id_col_1`` → ``ID``)
like ``generate_fold`` — do not invent stratification labels.

CLI mirrors ``generate_fold``: ``--id-csv``, ``--prepare-strat``, ``--outdir``.
This stub warns and raises without writing ``stratification.csv``.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path


def run_generate_stratification(
    id_csv: Path,
    prepare_strat_csv: Path,
    *,
    outdir: Path,
) -> Path:
    """Stub for ``{outdir}/stratification.csv`` generation.

    When implemented, resolve prepare_strat identificators through
    ``run_id_rule([identificator], id_csv, id_col_1=column, id_col_2="ID")``
    (same pattern as ``generate_fold``), then write ``ID`` plus stratification
    columns. Does not invent data; currently always warns and raises.
    """
    _ = (Path(id_csv), Path(prepare_strat_csv), Path(outdir))
    warnings.warn("Not implemented", UserWarning, stacklevel=2)
    raise NotImplementedError("Not implemented")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "generate_stratification: ID.csv + prepare_strat.csv → "
            "stratification.csv (stub; uses id_rule when implemented)"
        )
    )
    p.add_argument("--id-csv", required=True, type=Path)
    p.add_argument("--prepare-strat", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        print(
            run_generate_stratification(
                args.id_csv, args.prepare_strat, outdir=args.outdir
            )
        )
    except NotImplementedError:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
