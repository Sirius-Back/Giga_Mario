"""Auto-generated /split runner — VGAE direct on ready_legnet panel."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.split import run_split
from src.pipeline.split_predict import run_split_predict

DATA = "legnet"
SPLIT = "vgae"
MODE = "direct"

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "ready_legnet"
GRAPH = ROOT / "runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph"
OUT = ROOT / "VGAE" / "stage1_region_k5_split_materialize"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Prefer already-trained split.csv from VGAE/stage1_region_k5 when present
    trained = ROOT / "VGAE" / "stage1_region_k5" / "split.csv"
    if trained.is_file():
        split_csv = trained
    else:
        split_csv = run_split_predict(
            outdir=OUT,
            type=SPLIT,
            seed=42,
            id_csv=PANEL / "ID.csv",
            fold_csv=PANEL / "fold.csv" if (PANEL / "fold.csv").is_file() else None,
            marked_fasta=PANEL / "MARKED",
            ratios=(3.0, 1.0, 1.0),
            kmer_size=5,
            vgae_graph_dir=GRAPH,
        )
    run_split(
        split_csv,
        parsed_target=PANEL,
        parsed_data=PANEL,
        outdir=OUT,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL / "ID.csv",
    )


if __name__ == "__main__":
    main()
