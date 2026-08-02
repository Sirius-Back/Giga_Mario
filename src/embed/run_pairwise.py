"""CLI: pairwise LegNet embed-store comparison (no L(τ)).

Example::

  python -m src.embed.run_pairwise \\
    --embed-root results/embed_legnet \\
    --out results/embed_legnet/pairwise \\
    --layers pooled,stage1_2 \\
    --max-n 8192 --rdm-n 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embed.pairwise import DEFAULT_LAYERS, run_pairwise_compare

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embed-root",
        type=Path,
        default=ROOT / "results" / "embed_legnet",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "embed_legnet" / "pairwise",
    )
    p.add_argument("--layers", type=str, default=",".join(DEFAULT_LAYERS))
    p.add_argument(
        "--role",
        choices=("all", "test", "test_either"),
        default="all",
        help="ID set for alignment (default all: panel overlap; test∩test "
        "across splits is often empty)",
    )
    p.add_argument("--max-n", type=int, default=8192)
    p.add_argument("--rdm-n", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--loo-fold",
        type=int,
        default=0,
        help="Keep only this LOO fold index (default 0). Use --all-loo-folds "
        "to include every fold.",
    )
    p.add_argument(
        "--all-loo-folds",
        action="store_true",
        help="Include all LOO folds (overrides --loo-fold).",
    )
    args = p.parse_args(argv)

    layers = tuple(s.strip() for s in args.layers.split(",") if s.strip())
    loo_fold = None if args.all_loo_folds else int(args.loo_fold)
    run_pairwise_compare(
        args.embed_root,
        args.out,
        layers=layers,
        role=args.role,
        max_n=int(args.max_n),
        rdm_n=int(args.rdm_n),
        seed=int(args.seed),
        loo_fold=loo_fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
