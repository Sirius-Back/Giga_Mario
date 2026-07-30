"""Continue run8_2mer_caduceus from finished k-mer split → Caduceus train.

Assumes ``runs/run8_2mer_caduceus/{split.csv,SPLIT}`` and mice ZSV trees already
exist (panel ``ready_caduceus``). Skips k-mer feature recompute.

Default ``n_devices=2`` on GPUs 0,1 (user Locked).

Launch::

  CUDA_VISIBLE_DEVICES=0,1 conda run -n caduceus_env --no-capture-output \\
    python -m src.runs.run8_2mer_caduceus.continue_from_split
"""
from __future__ import annotations

import json
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


def _direct_final_ok() -> bool:
    weights = OUT_ROOT / "direct" / "final_model" / "model.safetensors"
    return weights.is_file() and weights.stat().st_size > 0


def _adv_split_train_dir(adv_split_root: Path) -> Path | None:
    """Return existing fold dir: legacy SPLIT/train or pipeline SPLIT/FASTA/TRAIN."""
    legacy = adv_split_root / "train"
    if legacy.is_dir():
        return legacy
    fasta_train = adv_split_root / "FASTA" / "TRAIN"
    if fasta_train.is_dir():
        return fasta_train
    return None


def _clean_failed_adv_train(train_dir: Path) -> None:
    """Drop failed/empty train weights/logs; keep caduceus_input if present."""
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


def _caduceus_input_ready(adv_input: Path) -> bool:
    """True when train/val/test labels+sequences exist under caduceus_input."""
    for fold in ("train", "val", "test"):
        if not (adv_input / fold / "labels.tsv").is_file():
            return False
        if not (adv_input / fold / "sequences").is_dir():
            return False
    return True


def _resolve_adv_train_folders(adv_split_root: Path, adv_input: Path) -> Path:
    """Reuse complete caduceus_input; otherwise rebuild from SPLIT."""
    if _caduceus_input_ready(adv_input):
        print(f"keeping existing caduceus_input at {adv_input}", flush=True)
        return adv_input
    if adv_input.exists():
        print(
            f"incomplete caduceus_input — removing for rebuild: {adv_input}",
            flush=True,
        )
        shutil.rmtree(adv_input)
    return adv_split_root


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

    if skip_direct and not _direct_final_ok():
        raise RuntimeError(
            f"skip_direct=true but missing {OUT_ROOT / 'direct' / 'final_model' / 'model.safetensors'}"
        )
    if not skip_direct and _direct_final_ok():
        # Resume-friendly: do not wipe a completed direct train unless forced.
        print(
            "direct/final_model already present — treating as skip_direct "
            "(pass force_direct=true to retrain)",
            flush=True,
        )
        skip_direct = True

    print(
        f"continue_from_split run_id={RUN_ID} split_csv={split_csv} "
        f"skip_direct={skip_direct} n_devices={n_devices} epochs={epochs} "
        f"min_epochs={min_epochs} patience={patience} batch_size={batch} "
        f"max_length={max_length}",
        flush=True,
    )

    if not skip_direct:
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
    else:
        print("skip_direct: using existing direct/final_model + ZSV", flush=True)

    adv_root = OUT_ROOT / "adversarial"
    adv_split_root = adv_root / "SPLIT"
    adv_input = adv_root / "train" / "caduceus_input"
    adv_split_csv = adv_root / "split.csv"
    adv_predict = adv_root / "PREDICT"
    adv_parsed = adv_root / "PARSED"
    split_train = _adv_split_train_dir(adv_split_root)
    # Resume when skip_adv_setup OR split.csv already exists.
    # Full wipe only when adversarial/split.csv is missing.
    if adv_split_csv.is_file():
        # split.csv present: reuse or rebuild SPLIT only (never rmtree adversarial/)
        if split_train is not None:
            print(
                f"reusing existing SPLIT train at {split_train} "
                f"(skip_adv_setup={skip_adv_setup})",
                flush=True,
            )
        elif adv_predict.is_dir() and adv_parsed.is_dir():
            print(
                "adversarial/split.csv + PREDICT present but SPLIT/train missing — "
                "rebuilding SPLIT only (no rmtree of adversarial/)",
                flush=True,
            )
            adv_split_root = run_split(
                adv_split_csv,
                parsed_target=adv_predict,
                parsed_data=adv_parsed,
                outdir=adv_root,
                strategy="traintestval",
                intersect_allow=True,
                id_csv=PANEL_ROOT / "ID.csv",
            )
            split_train = _adv_split_train_dir(Path(adv_split_root))
            if split_train is None:
                raise RuntimeError(
                    f"run_split completed but no TRAIN fold under {adv_split_root}"
                )
        else:
            raise RuntimeError(
                "adversarial/split.csv present but cannot rebuild SPLIT: "
                f"missing PREDICT/PARSED under {adv_root}"
            )
        if (adv_input / "train").is_dir() and not _caduceus_input_ready(adv_input):
            print(
                f"caduceus_input present but incomplete at {adv_input}",
                flush=True,
            )
        _clean_failed_adv_train(adv_root / "train")
    else:
        # split.csv missing → full rebuild
        if skip_adv_setup:
            print(
                "skip_adv_setup=true but adversarial/split.csv missing — "
                "falling back to full adversarial rebuild",
                flush=True,
            )
        if adv_root.exists():
            print(f"full adversarial rebuild: rmtree {adv_root}", flush=True)
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

    adv_train_folders = _resolve_adv_train_folders(Path(adv_split_root), adv_input)
    adv_weights = adv_root / "train" / "final_model" / "model.safetensors"
    if adv_weights.is_file() and adv_weights.stat().st_size > 0:
        print("adversarial final_model present — skip adv train", flush=True)
    else:
        run_train(
            model="caduceus",
            type="classification",
            folders=adv_train_folders,
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
