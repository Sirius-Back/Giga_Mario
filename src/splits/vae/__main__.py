"""CLI: ``python -m src.splits.vae …``"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MLP-VAE k-mer split (no GCN)")
    p.add_argument("--out", type=Path, required=True, help="Output under VAE/…")
    p.add_argument(
        "--features",
        type=Path,
        default=Path("runs_unif/legnet/run11_legnet_kmer_k4/feature_table.csv"),
    )
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda:N | auto")
    p.add_argument(
        "--prefer-gpu",
        action="store_true",
        help="Prefer free GPU over CPU when device is auto",
    )
    p.add_argument("--min-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--wait-poll-sec", type=float, default=600.0)
    p.add_argument("--peak-ram-gib", type=float, default=8.0)
    p.add_argument(
        "--project-dim",
        type=int,
        default=None,
        help="Optional seeded projection of k-mer dims (e.g. 2048 for k=7)",
    )
    p.add_argument("--project-seed", type=int, default=42)
    p.add_argument(
        "--keep-memmap",
        action="store_true",
        help="Keep full feature matrix as disk memmap (required for k=7 16384-d)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Microbatch size for GPU train (e.g. 2048 for full 16384-d)",
    )
    p.add_argument("--skip-train-viz", action="store_true")
    p.add_argument(
        "--source-label",
        type=str,
        default="run11_legnet_kmer_k4",
    )
    args = p.parse_args(argv)

    prefer_cpu = not bool(args.prefer_gpu)
    device = args.device
    if device == "auto":
        device = None

    # k>=7 dense panels need more host RAM for pack/project unless already packed
    peak = float(args.peak_ram_gib)
    if args.k >= 7 and peak < 16.0:
        peak = 16.0
    if args.keep_memmap and peak < 12.0:
        peak = 12.0

    from src.splits.vae.train import run_vae_train

    meta = run_vae_train(
        args.out / "pack",
        args.out,
        features_path=args.features,
        k=args.k,
        seed=args.seed,
        device=device,
        prefer_cpu=prefer_cpu,
        min_epochs=args.min_epochs,
        patience=args.patience,
        max_epochs=args.max_epochs,
        wait_poll_sec=args.wait_poll_sec,
        peak_ram_gib=peak,
        source_label=args.source_label,
        project_dim=args.project_dim,
        project_seed=args.project_seed,
        keep_memmap=bool(args.keep_memmap),
        batch_size=args.batch_size,
    )
    import json

    print(json.dumps(meta, indent=2, default=str), flush=True)

    if not args.skip_train_viz:
        try:
            from src.train_viz.viz import main as viz_main

            fig_dir = args.out / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            code = viz_main(
                [
                    str(args.out / "logs"),
                    "-o",
                    str(fig_dir),
                    "--model",
                    f"mlp_vae_k{args.k}",
                ]
            )
            if int(code or 0) != 0:
                print(f"[vae] train-viz exited code={code}", flush=True)
        except SystemExit as exc:
            if int(getattr(exc, "code", 1) or 1) != 0:
                print(f"[vae] train-viz exited non-zero: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[vae] train-viz failed (non-fatal): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
