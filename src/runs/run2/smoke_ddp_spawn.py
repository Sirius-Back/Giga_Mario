"""Smoke: 1-epoch LegNet on 2 GPUs (ddp_spawn)."""
from __future__ import annotations

from pathlib import Path

from src.legnet import run

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    return run(
        data_path=ROOT / "runs/run2/legnet_input/all.tsv",
        out_dir=ROOT / "runs/run2/direct_d2s",
        vendor=ROOT / "software/human_legnet",
        epochs=1,
        train_batch_size=8192,
        valid_batch_size=8192,
        num_workers=0,
        seed=42,
        device=0,
        n_devices=2,
        demo=True,
        use_shift=False,
        reverse_augment=False,
        use_reverse_channel=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
