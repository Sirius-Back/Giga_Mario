"""CLI: ``python -m src.splits.vgae …``"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="VGAE pangenome split (Stage1/Stage2)")
    p.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p.add_argument("--out", type=Path, required=True, help="Output under VGAE/…")
    p.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph"),
    )
    p.add_argument("--marked-dir", type=Path, default=Path("ready_legnet/MARKED"))
    p.add_argument("--ids-file", type=Path, default=None, help="Optional ids.txt override")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-ids", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--wait-poll-sec", type=float, default=600.0)
    p.add_argument(
        "--loss-mode",
        type=str,
        default="legacy",
        choices=("legacy", "homology_first"),
        help="legacy = unnormalized recon+KL+L_hom; homology_first = EMA/anneal/Gumbel",
    )
    p.add_argument("--lambda-hom", type=float, default=None)
    p.add_argument("--lambda-size", type=float, default=None)
    p.add_argument("--lambda-para", type=float, default=None)
    p.add_argument("--lambda-ortho", type=float, default=None)
    p.add_argument("--alpha-recon", type=float, default=None)
    p.add_argument("--beta-kl-max", type=float, default=None)
    p.add_argument("--skip-train-viz", action="store_true")
    args = p.parse_args(argv)

    train_extra: dict = {"loss_mode": args.loss_mode}
    if args.lambda_hom is not None:
        train_extra["lambda_hom"] = float(args.lambda_hom)
    if args.lambda_size is not None:
        train_extra["lambda_size"] = float(args.lambda_size)
    if args.lambda_para is not None:
        train_extra["lambda_para"] = float(args.lambda_para)
    if args.lambda_ortho is not None:
        train_extra["lambda_ortho"] = float(args.lambda_ortho)
    if args.alpha_recon is not None:
        train_extra["alpha_recon"] = float(args.alpha_recon)
    if args.beta_kl_max is not None:
        train_extra["beta_kl_max"] = float(args.beta_kl_max)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == 1:
        from src.splits.vgae.split_assign import run_vgae_split_assign

        existing_pack = out / "pack"
        pack_dir = (
            existing_pack
            if (existing_pack / "feature_meta.json").is_file()
            else None
        )
        meta = run_vgae_split_assign(
            outdir=out,
            graph_dir=args.graph_dir if pack_dir is None else None,
            marked_dir=args.marked_dir if pack_dir is None else None,
            pack_dir=pack_dir,
            seed=args.seed,
            k=args.k,
            max_ids=args.max_ids,
            device=args.device,
            min_epochs=args.min_epochs,
            patience=args.patience,
            max_epochs=args.max_epochs,
            wait_poll_sec=args.wait_poll_sec,
            **train_extra,
        )
    else:
        from src.splits.vgae.stage2 import run_stage2_hash_vgae

        ids_file = args.ids_file
        if ids_file is None:
            ids_file = Path(args.graph_dir)
            if ids_file.name != "graph":
                ids_file = ids_file / "graph"
            ids_file = ids_file / "ids.txt"
        region_ids = [
            ln.strip()
            for ln in ids_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        meta = run_stage2_hash_vgae(
            out_dir=out,
            marked_dir=args.marked_dir,
            region_ids=region_ids,
            k=args.k,
            seed=args.seed,
            max_ids=args.max_ids,
            device=args.device,
            min_epochs=args.min_epochs,
            patience=args.patience,
            max_epochs=args.max_epochs,
            wait_poll_sec=args.wait_poll_sec,
            **train_extra,
        )

    print(json_dumps(meta), flush=True)

    if not args.skip_train_viz:
        try:
            from src.train_viz.viz import main as viz_main

            fig_dir = out / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            code = viz_main(
                [
                    str(out / "logs"),
                    "-o",
                    str(fig_dir),
                    "--model",
                    f"vgae_stage{args.stage}",
                ]
            )
            if int(code or 0) != 0:
                print(f"[vgae] train-viz exited code={code}", flush=True)
        except SystemExit as exc:
            if int(getattr(exc, "code", 1) or 1) != 0:
                print(f"[vgae] train-viz exited non-zero: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 — viz is best-effort after train
            print(f"[vgae] train-viz failed (non-fatal): {exc}", flush=True)

    return 0


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
