"""Continue run13_7mer_legnet from existing 7-mer split → LegNet + adversarial.

Assumes ``runs/run13_7mer_legnet/{split.csv,SPLIT,legnet_input/all.tsv}`` and mice
ZSV trees already exist.

Launch (after GPUs 2,3 free)::

  CUDA_VISIBLE_DEVICES=2,3 conda run -n legnet --no-capture-output \\
    python -m src.runs.run13_7mer_legnet.continue_from_split
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run13_7mer_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
# Default 1 GPU — 2-GPU LegNet ddp_spawn hangs on this host (run2/run4/run7).
N_DEVICES = 1
BATCH_SIZE = 8192
NUM_WORKERS = 8
SEED = 42
RATIOS = (1, 1, 3)


def _require(path: Path, kind: str = "path") -> Path:
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Missing required dir: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    min_epochs = MIN_EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    n_devices = N_DEVICES
    for tok in list(argv):
        if tok.startswith("batch_size="):
            batch = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("epochs="):
            epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("min_epochs="):
            min_epochs = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("early_stopping_patience="):
            patience = int(tok.split("=", 1)[1])
            argv.remove(tok)
        elif tok.startswith("n_devices="):
            n_devices = int(tok.split("=", 1)[1])
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    _require(OUT_ROOT / "SPLIT", "dir")
    tsv = _require(OUT_ROOT / "legnet_input" / "all.tsv", "file")
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")
    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "fold.csv", "file")

    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train

    print(
        f"continue_from_split: tsv={tsv} epochs={epochs} min_epochs={min_epochs} "
        f"patience={patience} n_devices={n_devices} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    direct_out = OUT_ROOT / "direct"
    if direct_out.exists():
        shutil.rmtree(direct_out)

    run_train(
        model="legnet",
        type="regression",
        folders=tsv,
        outdir=direct_out,
        strategy="kmer",
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        seed=SEED,
        n_devices=n_devices,
        num_workers=NUM_WORKERS,
        legnet_demo=True,
        zsv_root=OUT_ROOT,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
    )

    try:
        from src.pipeline.pipeline_viz import run_pipeline_viz_auto

        viz = run_pipeline_viz_auto(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            train_dir=direct_out,
            run_id=RUN_ID,
            seed=SEED,
            plot_train=True,
            plot_sbs=True,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
        print(f"pipeline_viz[direct] status={viz.get('status')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: pipeline_viz[direct] skipped: {type(exc).__name__}: {exc}",
            flush=True,
        )

    adv_root = OUT_ROOT / "adversarial"
    if adv_root.exists():
        shutil.rmtree(adv_root)
    run_adversarial(
        outdir_new=adv_root,
        split_csv=split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        intersect_allow=True,
    )
    adv_split = run_split_predict(
        outdir=adv_root,
        type="random",
        seed=SEED + 1,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
    )
    apply_fold_class_targets(
        predict_root=adv_root / "PREDICT",
        label_split_csv=split_csv,
    )
    run_split(
        adv_split,
        parsed_target=adv_root / "PREDICT",
        parsed_data=adv_root / "PARSED",
        outdir=adv_root,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL_ROOT / "ID.csv",
    )
    adv_tsv = build_legnet_tsv(
        split_root=adv_root / "SPLIT",
        out_tsv=adv_root / "legnet_input" / "all.tsv",
    )
    run_train(
        model="legnet",
        type="classification",
        folders=adv_tsv,
        outdir=adv_root / "train",
        strategy="random",
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        seed=SEED,
        n_devices=n_devices,
        num_workers=NUM_WORKERS,
        legnet_demo=True,
        zsv_root=adv_root,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
    )

    try:
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        mon = refresh_pipeline_monitors(
            OUT_ROOT, run_id=RUN_ID, include_split_compare=True
        )
        print(f"pipeline_monitors status={mon.get('status')}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: pipeline_monitors skipped: {type(exc).__name__}: {exc}",
            flush=True,
        )

    (OUT_ROOT / "pipeline_done.json").write_text(
        '{"status":"COMPLETED","run_id":"%s"}\n' % RUN_ID, encoding="utf-8"
    )
    print(f"run13_7mer_legnet continue COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
