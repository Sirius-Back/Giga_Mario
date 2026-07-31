"""Align orthoparagroups FASTAs with MAFFT and score consensus rates."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from src.homology.align_consensus import (
    apply_size_norm,
    discover_fna,
    fit_size_norm_models,
    models_to_jsonable,
    process_cluster,
)


def _worker(args: tuple[str, str, str, int]) -> tuple[str, str | None, str | None]:
    """Return (stem, metrics_tsv_path_or_none, error_or_none)."""
    fna_s, outdir_s, mafft_bin, threads = args
    fna = Path(fna_s)
    outdir = Path(outdir_s)
    try:
        stem, metrics = process_cluster(fna, outdir, mafft_bin=mafft_bin, threads=threads)
        raw_path = outdir / "metrics_raw" / f"{stem}.pos.tsv.gz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(raw_path, sep="\t", index=False, compression="gzip")
        return stem, str(raw_path), None
    except Exception as exc:  # noqa: BLE001 — collect per-cluster failures
        return fna.stem, None, f"{type(exc).__name__}: {exc}"


def _cluster_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "cluster": df["cluster"].iloc[0],
        "aln_length": int(len(df)),
        "n_seqs": int(df["n_seqs"].iloc[0]),
        "n_orthologs": int(df["n_orthologs"].iloc[0]),
        "n_paralogs": int(df["n_paralogs"].iloc[0]),
        "mean_overall_consensus_rate": float(df["overall_consensus_rate"].mean(skipna=True)),
        "mean_orthologs_consensus_rate": float(df["orthologs_consensus_rate"].mean(skipna=True)),
        "mean_paralogs_consensus_rate": float(df["paralogs_consensus_rate"].mean(skipna=True)),
        "mean_overall_consensus_rate_norm_residual": float(
            df["overall_consensus_rate_norm_residual"].mean(skipna=True)
        ),
        "mean_orthologs_consensus_rate_norm_residual": float(
            df["orthologs_consensus_rate_norm_residual"].mean(skipna=True)
        ),
        "mean_paralogs_consensus_rate_norm_residual": float(
            df["paralogs_consensus_rate_norm_residual"].mean(skipna=True)
        ),
        "mean_gap_fraction": float(df["gap_fraction"].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", type=Path, default=Path("mag/orthoparagroups"))
    p.add_argument("--outdir", type=Path, default=Path("mag/orthoparagroups_aligned"))
    p.add_argument(
        "--mafft",
        type=str,
        default=os.environ.get(
            "MAFFT_BIN",
            str(Path.home() / "miniconda3/envs/bio_tools/bin/mafft"),
        ),
    )
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) // 2)))
    p.add_argument("--mafft-threads", type=int, default=1)
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="If >0, only first N clusters (smoke)")
    p.add_argument("--skip-align", action="store_true", help="Reuse existing *.aln.fa + metrics_raw")
    args = p.parse_args(argv)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fnas = discover_fna(args.indir)
    if args.limit > 0:
        fnas = fnas[: args.limit]

    mafft_bin = args.mafft
    if not args.skip_align and not Path(mafft_bin).is_file():
        raise FileNotFoundError(f"mafft binary not found: {mafft_bin}")

    print(f"[align] clusters={len(fnas)} workers={args.workers} outdir={outdir}", flush=True)

    position_tables: dict[str, pd.DataFrame] = {}
    errors: list[tuple[str, str]] = []

    if args.skip_align:
        raw_dir = outdir / "metrics_raw"
        for path in sorted(raw_dir.glob("cluster_*.pos.tsv.gz")):
            df = pd.read_csv(path, sep="\t")
            position_tables[df["cluster"].iloc[0]] = df
        if not position_tables:
            raise FileNotFoundError(f"No metrics_raw under {raw_dir}")
    else:
        jobs = [(str(f), str(outdir), mafft_bin, args.mafft_threads) for f in fnas]
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, job) for job in jobs]
            for fut in as_completed(futs):
                stem, raw_path, err = fut.result()
                done += 1
                if err:
                    errors.append((stem, err))
                    print(f"[align] FAIL {stem}: {err}", flush=True)
                else:
                    assert raw_path is not None
                    df = pd.read_csv(raw_path, sep="\t")
                    position_tables[stem] = df
                if done % 100 == 0 or done == len(jobs):
                    print(f"[align] progress {done}/{len(jobs)} ok={len(position_tables)} fail={len(errors)}", flush=True)

    if len(position_tables) < 4:
        raise RuntimeError(f"Too few successful alignments ({len(position_tables)}) to fit models")

    print(f"[norm] fitting size models on train_fraction={args.train_fraction} seed={args.seed}", flush=True)
    models, meta = fit_size_norm_models(
        position_tables,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    model_path = outdir / "normalization_model.json"
    model_path.write_text(
        json.dumps(models_to_jsonable(models, meta), indent=2) + "\n",
        encoding="utf-8",
    )

    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for stem, raw in sorted(position_tables.items()):
        normed = apply_size_norm(raw, models)
        out_path = metrics_dir / f"{stem}.pos.tsv.gz"
        normed.to_csv(out_path, sep="\t", index=False, compression="gzip")
        summaries.append(_cluster_summary(normed))

    summary_df = pd.DataFrame(summaries).sort_values("cluster")
    summary_path = outdir / "cluster_consensus_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    err_path = outdir / "align_errors.tsv"
    if errors:
        pd.DataFrame(errors, columns=["cluster", "error"]).to_csv(err_path, sep="\t", index=False)
    elif err_path.exists():
        err_path.unlink()

    manifest = {
        "n_input": len(fnas),
        "n_aligned": len(position_tables),
        "n_failed": len(errors),
        "outdir": str(outdir),
        "mafft": mafft_bin,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "model": {k: models[k].__dict__ for k in ("overall", "orthologs", "paralogs")},
        "held_out_n": len(meta["held_out_clusters"]),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[done] aligned={len(position_tables)} failed={len(errors)} "
        f"summary={summary_path} model={model_path}",
        flush=True,
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
