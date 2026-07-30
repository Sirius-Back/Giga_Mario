"""Continue run12_4mer_caduceus from 4-mer split → Caduceus + adversarial.

Resumes from saved artifacts: skips direct/adv when ``final_model`` weights exist;
keeps ``caduceus_input/`` when re-training after a failed run.

Launch (after GPUs 0,1 free)::

  CUDA_VISIBLE_DEVICES=0,1 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run12_4mer_caduceus.continue_from_split
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run12_4mer_caduceus"
PANEL_ROOT = ROOT / "ready_caduceus"
OUT_ROOT = ROOT / "runs" / RUN_ID
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


def _final_ok(train_dir: Path) -> bool:
    w = train_dir / "final_model" / "model.safetensors"
    return w.is_file() and w.stat().st_size > 0


def _clean_train_keep_input(train_dir: Path) -> None:
    """Remove failed train artifacts but keep ``caduceus_input/`` if present."""
    if not train_dir.is_dir():
        return
    for name in (
        "final_model",
        "best_model",
        "checkpoints",
        "logs",
        "tensorboard",
        "figures",
        "train_time.json",
        "run_config.json",
    ):
        p = train_dir / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    batch = BATCH_SIZE
    epochs = EPOCHS
    min_epochs = MIN_EPOCHS
    patience = EARLY_STOPPING_PATIENCE
    n_devices = N_DEVICES
    max_length = MAX_LENGTH
    skip_direct = False
    skip_adv_setup = False
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
        elif tok in {"skip_direct=true", "--skip-direct"}:
            skip_direct = True
            argv.remove(tok)
        elif tok in {"skip_adv_setup=true", "--skip-adv-setup"}:
            skip_adv_setup = True
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

    from src.pipeline.adversarial import apply_fold_class_targets, run_adversarial
    from src.pipeline.pipeline_viz import run_pipeline_viz_auto
    from src.pipeline.split import run_split
    from src.pipeline.split_predict import run_split_predict
    from src.pipeline.train import run_train

    if _final_ok(OUT_ROOT / "direct"):
        skip_direct = True
    if skip_direct and not _final_ok(OUT_ROOT / "direct"):
        raise RuntimeError("skip_direct but direct/final_model missing")

    print(
        f"continue_from_split run_id={RUN_ID} skip_direct={skip_direct} "
        f"skip_adv_setup={skip_adv_setup} n_devices={n_devices} "
        f"epochs={epochs} min_epochs={min_epochs} patience={patience} "
        f"batch_size={batch} max_length={max_length}",
        flush=True,
    )

    if not skip_direct:
        _clean_train_keep_input(OUT_ROOT / "direct")
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
        try:
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
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: direct pipeline_viz skipped: {type(exc).__name__}: {exc}",
                flush=True,
            )
    else:
        print("skip_direct: using existing direct/final_model", flush=True)

    adv_root = OUT_ROOT / "adversarial"
    adv_split_root = adv_root / "SPLIT"
    if skip_adv_setup or (
        (adv_split_root / "train").is_dir() and (adv_root / "split.csv").is_file()
    ):
        if not ((adv_split_root / "train").is_dir() and (adv_root / "split.csv").is_file()):
            raise RuntimeError("skip_adv_setup but adversarial SPLIT missing")
        print(f"skip_adv_setup: reusing {adv_root}", flush=True)
        _clean_train_keep_input(adv_root / "train")
    else:
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

    if _final_ok(adv_root / "train"):
        print("adversarial final_model already present — skip adv train", flush=True)
    else:
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
        try:
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
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: adversarial pipeline_viz skipped: {type(exc).__name__}: {exc}",
                flush=True,
            )

    done = {
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "direct": str(OUT_ROOT / "direct"),
        "adversarial": str(adv_root / "train"),
        "skip_direct": skip_direct,
        "epochs": epochs,
        "min_epochs": min_epochs,
        "early_stopping_patience": patience,
        "n_devices": n_devices,
        "batch_size": batch,
    }
    (OUT_ROOT / "pipeline_done.json").write_text(
        json.dumps(done, indent=2) + "\n", encoding="utf-8"
    )
    print("continue_from_split COMPLETED", flush=True)
    print(f"pipeline_done={OUT_ROOT / 'pipeline_done.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
