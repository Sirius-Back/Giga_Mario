"""run17 pipeline: ready_legnet → pangenome CDS split → LegNet direct + adversarial.

Scripts: ``src/runs/run17_pangenome_CDS_legnet/``; artifacts: ``runs/run17_pangenome_CDS_legnet/``.

- Direct: pangenome C++ contingency on CDS window 0..100 (adapt from ``./raw``)
- Train: LegNet regression, 4 GPUs, epochs 25–50 with early stop, best→final
- ZSV: mice genome ``GCF_000001635.27``
- Adversarial: random split + fold-class PREDICT + early stop

Staged launch (preferred)::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.run_split_cpu
  # waits for GPUs 0–3 free, then:
  CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n legnet --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.continue_from_split

Or end-to-end (CPU split first, then GPU wait + train)::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.pipeline_ready_legnet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run17_pangenome_CDS_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    skip_split = False
    for tok in list(argv):
        if tok in {"skip_split=true", "--skip-split"}:
            skip_split = True
            argv.remove(tok)

    split_argv = [
        t
        for t in argv
        if not t.startswith(
            (
                "epochs=",
                "batch_size=",
                "n_devices=",
                "min_epochs=",
                "early_stopping_patience=",
                "skip_wait=",
            )
        )
        and t not in {"--skip-wait"}
    ]
    train_argv = [
        t
        for t in argv
        if t.startswith(
            (
                "epochs=",
                "batch_size=",
                "n_devices=",
                "min_epochs=",
                "early_stopping_patience=",
                "skip_wait=",
            )
        )
        or t == "--skip-wait"
    ]

    if not skip_split and not (OUT_ROOT / "split_cpu_done.json").is_file():
        from src.runs.run17_pangenome_CDS_legnet.run_split_cpu import main as split_main

        rc = split_main(split_argv)
        if rc != 0:
            return rc
    else:
        print(
            f"skip_split or split_cpu_done present → {OUT_ROOT / 'split_cpu_done.json'}",
            flush=True,
        )

    # Default train overrides matching user Locked settings.
    if not any(t.startswith("n_devices=") for t in train_argv):
        train_argv = ["n_devices=4", *train_argv]
    if not any(t.startswith("min_epochs=") for t in train_argv):
        train_argv = ["min_epochs=25", *train_argv]
    if not any(t.startswith("epochs=") for t in train_argv):
        train_argv = ["epochs=50", *train_argv]
    if not any(t.startswith("early_stopping_patience=") for t in train_argv):
        train_argv = ["early_stopping_patience=10", *train_argv]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    from src.runs.run17_pangenome_CDS_legnet.continue_from_split import main as train_main

    return train_main(train_argv)


if __name__ == "__main__":
    raise SystemExit(main())
