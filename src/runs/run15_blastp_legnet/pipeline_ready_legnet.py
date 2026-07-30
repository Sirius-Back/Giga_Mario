"""run15 pipeline: ready_legnet → BLASTP split → LegNet direct.

Scripts: ``src/runs/run15_blastp_legnet/``; artifacts: ``runs/run15_blastp_legnet/``.

- Direct: BLASTP CDS homology connected components, LegNet regression
- ZSV: mice genome ``GCF_000001635.27``
- Adversarial: out of scope for this runner (see plan)

Staged launch (preferred for long BLASTP)::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run15_blastp_legnet.run_split_cpu
  CUDA_VISIBLE_DEVICES=0 conda run -n legnet --no-capture-output \\
    python -m src.runs.run15_blastp_legnet.continue_from_split

Or end-to-end::

  CUDA_VISIBLE_DEVICES=0 conda run -n legnet --no-capture-output \\
    python -m src.runs.run15_blastp_legnet.pipeline_ready_legnet
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run15_blastp_legnet"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.runs.run15_blastp_legnet.run_split_cpu import main as split_main
    from src.runs.run15_blastp_legnet.continue_from_split import main as train_main

    split_argv = [t for t in argv if not t.startswith("epochs=") and not t.startswith("batch_size=")
                  and not t.startswith("n_devices=") and not t.startswith("min_epochs=")
                  and not t.startswith("early_stopping_patience=")]
    train_argv = [t for t in argv if t.startswith("epochs=") or t.startswith("batch_size=")
                  or t.startswith("n_devices=") or t.startswith("min_epochs=")
                  or t.startswith("early_stopping_patience=")]

    rc = split_main(split_argv)
    if rc != 0:
        return rc
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return train_main(train_argv)


if __name__ == "__main__":
    raise SystemExit(main())
