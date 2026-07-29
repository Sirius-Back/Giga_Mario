"""Continue run8_2mer_caduceus from finished k-mer split → Caduceus train.

Assumes ``runs/run8_2mer_caduceus/{split.csv,SPLIT}`` and mice ZSV trees already
exist (panel ``ready_caduceus``). Skips k-mer feature recompute.

Default ``n_devices=2`` on GPUs 0,1 (user Locked).

Launch::

  CUDA_VISIBLE_DEVICES=0,1 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run8_2mer_caduceus.continue_from_split
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run8_2mer_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / "run8_2mer_caduceus"
EPOCHS = 50
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10
N_DEVICES = 2
BATCH_SIZE = 480
MAX_LENGTH = 208
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
    max_length = MAX_LENGTH
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
        elif tok.startswith("max_length="):
            max_length = int(tok.split("=", 1)[1])
            argv.remove(tok)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    split_csv = _require(OUT_ROOT / "split.csv", "file")
    split_root = _require(OUT_ROOT / "SPLIT", "dir")
    _require(OUT_ROOT / "PARSED" / "zero-shot-validation", "dir")
    _require(OUT_ROOT / "PREDICT" / "zero-shot-validation", "dir")
    _require(PANEL_ROOT / "ID.csv", "file")
    _require(PANEL_ROOT / "fold.csv", "file")

    meta = OUT_ROOT / "kmer_split_meta.json"
    if meta.is_file():
        text = meta.read_text(encoding="utf-8")
        if "ready_legnet" in text and "ready_caduceus" not in text:
            raise RuntimeError(
                f"{meta} still references ready_legnet; re-run full "
                "pipeline_ready_caduceus to rebuild the k-mer split on "
                "ready_caduceus"
            )

    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train
    from src.pipeline.pipeline_viz import run_pipeline_viz_auto

    print(
        f"continue_from_split run_id={RUN_ID} split_csv={split_csv} "
        f"n_devices={n_devices} epochs={epochs} min_epochs={min_epochs} "
        f"patience={patience} batch_size={batch} max_length={max_length}",
        flush=True,
    )

    run_train(
        model="caduceus",
        type="regression",
        folders=split_root,
        outdir=OUT_ROOT / "direct",
        strategy="kmer",
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        max_length=max_length,
        seed=SEED,
        n_devices=n_devices,
        num_workers=NUM_WORKERS,
        legnet_demo=False,
        zsv_root=OUT_ROOT,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
    )
    run_pipeline_viz_auto(
        out_root=OUT_ROOT,
        panel_root=PANEL_ROOT,
        train_dir=OUT_ROOT / "direct",
        run_id=RUN_ID,
        seed=SEED,
        plot_train=True,
        plot_sbs=True,
        include_split_compare=True,
        viz_conda_env="caduceus_env",
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
    adv_seed = SEED + 1
    adv_split_csv = run_split_predict(
        outdir=adv_root,
        type="random",
        seed=adv_seed,
        id_csv=PANEL_ROOT / "ID.csv",
        fold_csv=PANEL_ROOT / "fold.csv",
        ratios=RATIOS,
    )
    apply_fold_class_targets(
        predict_root=adv_root / "PREDICT",
        label_split_csv=split_csv,
    )
    adv_split_root = run_split(
        adv_split_csv,
        parsed_target=adv_root / "PREDICT",
        parsed_data=adv_root / "PARSED",
        outdir=adv_root,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL_ROOT / "ID.csv",
    )
    run_train(
        model="caduceus",
        type="classification",
        folders=adv_split_root,
        outdir=adv_root / "train",
        strategy="random",
        smoke=False,
        epochs=epochs,
        batch_size=batch,
        max_length=max_length,
        seed=SEED,
        n_devices=n_devices,
        num_workers=NUM_WORKERS,
        legnet_demo=False,
        zsv_root=adv_root,
        eval_zsv=True,
        checkpoint_every_n_epochs=10,
        early_stopping_patience=patience,
        min_epochs=min_epochs,
    )
    run_pipeline_viz_auto(
        out_root=OUT_ROOT,
        panel_root=PANEL_ROOT,
        train_dir=adv_root / "train",
        run_id=RUN_ID,
        seed=SEED,
        plot_train=True,
        plot_sbs=False,
        include_split_compare=True,
        viz_conda_env="caduceus_env",
    )
    print("continue_from_split COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
