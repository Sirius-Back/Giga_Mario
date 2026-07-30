"""Continue run5 from existing hashFrag ``split.csv`` → materialize + LegNet + adversarial.

Skips BLAST / homologous-group rebuild. Uses project orthogonal assigner output.

Launch::

  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n legnet --no-capture-output \\
    python -m src.runs.run5.continue_from_split
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run5"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / "run5"
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
# 1-GPU default: ddp_spawn hangs after sanity on this host (see method-decision).
N_DEVICES = 1
BATCH_SIZE = 8192  # ≈ former 2048×4 global batch
NUM_WORKERS = 8
SEED = 42


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

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "PARSED", "dir")
    _require(PANEL_ROOT / "PREDICT", "dir")

    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.pipeline.train import run_train

    print(
        f"continue_from_split run5: split_csv={split_csv} "
        f"epochs={epochs} min_epochs={min_epochs} patience={patience} "
        f"n_devices={n_devices} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    split_root = OUT_ROOT / "SPLIT"
    train_fa = split_root / "FASTA" / "TRAIN"
    if train_fa.is_dir() and any(train_fa.iterdir()):
        print(f"reuse existing SPLIT: {split_root}", flush=True)
    else:
        print("materialize SPLIT/ …", flush=True)
        split_root = run_split(
            split_csv,
            parsed_target=PANEL_ROOT / "PREDICT",
            parsed_data=PANEL_ROOT / "PARSED",
            outdir=OUT_ROOT,
            strategy="traintestval",
            intersect_allow=True,
            id_csv=PANEL_ROOT / "ID.csv",
        )
        print(f"SPLIT ready: {split_root}", flush=True)

    tsv_path = OUT_ROOT / "legnet_input" / "all.tsv"
    if tsv_path.is_file() and tsv_path.stat().st_size > 0:
        tsv = tsv_path
        print(f"reuse legnet TSV: {tsv}", flush=True)
    else:
        tsv = build_legnet_tsv(split_root=split_root, out_tsv=tsv_path)
        print(f"legnet TSV: {tsv}", flush=True)

    direct_out = OUT_ROOT / "direct"
    if direct_out.exists():
        shutil.rmtree(direct_out)

    print("direct LegNet train …", flush=True)
    run_train(
        model="legnet",
        type="regression",
        folders=tsv,
        outdir=direct_out,
        strategy="hashfrag",
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

    print("adversarial copy + random split + fold-class …", flush=True)
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
    from src.pipeline.split_predict import run_split_predict

    adv_split = run_split_predict(
        outdir=adv_root,
        type="random",
        seed=SEED + 1,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=(1, 1, 3),
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
    print("adversarial LegNet train …", flush=True)
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
        from src.pipeline.pipeline_viz import run_pipeline_viz_auto
        from src.train_viz.train_monitor import refresh_pipeline_monitors

        run_pipeline_viz_auto(
            out_root=OUT_ROOT,
            panel_root=PANEL_ROOT,
            train_dir=direct_out,
            run_id=RUN_ID,
            seed=SEED,
            plot_train=True,
            plot_sbs=False,
            include_split_compare=True,
            viz_conda_env="caduceus_env",
        )
        refresh_pipeline_monitors(
            OUT_ROOT,
            run_id=RUN_ID,
            include_split_compare=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: viz stage failed: {exc}", flush=True)

    print(f"continue_from_split run5 complete → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
