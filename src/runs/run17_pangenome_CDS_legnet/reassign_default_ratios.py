"""Reassign run17 folds with Caduceus-default ratios (ratios=None).

The prior ``(1, 1, 3)`` train:test:val weights put ~273k regions in validation
(LegNet fold 2). Default weights are ~81% / 10% / 9% train/test/val.

Reuses the saved contingency graph (no k-mer rebuild). Rewrites assignment,
``split.csv``, rematerializes ``SPLIT/``, and rebuilds ``legnet_input/all.tsv``.

Launch::

  conda run -n legnet --no-capture-output \\
    python -m src.runs.run17_pangenome_CDS_legnet.reassign_default_ratios
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RUN_ID = "run17_pangenome_CDS_legnet"
PANEL_ROOT = ROOT / "ready_legnet"
OUT_ROOT = ROOT / "runs" / RUN_ID
SEED = 42
# Explicit None → Caduceus-aligned ~81/10/9 (see src.splits.common.train_test_val_weights).
RATIOS: tuple[float, float, float] | None = None


def main(argv: list[str] | None = None) -> int:
    del argv  # no CLI flags
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split
    from src.splits.pangenome import assign_from_contingency, load_contingency_graph
    from src.splits.sbs.assign import assignment_rows_to_split_csv, write_assignment_table

    graph_dir = OUT_ROOT / "graph"
    fold_csv = PANEL_ROOT / "fold.csv"
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"missing contingency graph dir: {graph_dir}")
    if not fold_csv.is_file():
        raise FileNotFoundError(f"missing fold.csv: {fold_csv}")

    # Preserve previous assignment for audit.
    for name in ("split.csv", "pangenome_assignment.csv"):
        src = OUT_ROOT / name
        if src.is_file():
            bak = OUT_ROOT / f"{name}.ratios_1_1_3.bak"
            if not bak.is_file():
                shutil.copy2(src, bak)
                print(f"backed up {src.name} → {bak.name}", flush=True)

    g = load_contingency_graph(graph_dir)
    ids = list(g["ids"])
    cluster_ids = list(g["cluster_ids"])
    print(
        f"loaded graph n={len(ids)} n_clusters={g['n_clusters']} ratios={RATIOS!r}",
        flush=True,
    )

    rows, assign_meta = assign_from_contingency(
        ids,
        cluster_ids,
        fold_csv=fold_csv,
        seed=SEED,
        ratios=RATIOS,
    )
    tt = Counter(r["train_test"] for r in rows)
    print(f"reassigned train_test counts: {dict(tt)}", flush=True)
    print(f"assign_meta n_clusters={assign_meta.get('n_clusters')}", flush=True)

    assign_path = write_assignment_table(rows, OUT_ROOT / "pangenome_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, OUT_ROOT)
    meta_path = OUT_ROOT / "pangenome_split_meta.json"
    prev_meta: dict = {}
    if meta_path.is_file():
        prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prev_meta.update(
        {
            "ratios": None,
            "ratios_note": "reassigned from (1,1,3) → default Caduceus ~81/10/9",
            "assign_meta": assign_meta,
            "assignment_csv": str(assign_path),
            "split_csv": str(split_csv),
            "reassign_seed": SEED,
        }
    )
    meta_path.write_text(json.dumps(prev_meta, indent=2) + "\n", encoding="utf-8")

    # Drop stale materialization before rebuild.
    split_dir = OUT_ROOT / "SPLIT"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    direct = OUT_ROOT / "direct"
    if direct.exists():
        shutil.rmtree(direct)

    split_root = run_split(
        split_csv,
        parsed_target=PANEL_ROOT / "PREDICT",
        parsed_data=PANEL_ROOT / "PARSED",
        outdir=OUT_ROOT,
        strategy="traintestval",
        intersect_allow=True,
        id_csv=PANEL_ROOT / "ID.csv",
    )
    tsv = build_legnet_tsv(
        split_root=split_root, out_tsv=OUT_ROOT / "legnet_input" / "all.tsv"
    )

    # Sanity: LegNet fold mapping TEST=1 VAL=2 TRAIN=3
    fold_counts: Counter[str] = Counter()
    with tsv.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        fi = header.index("fold")
        for line in fh:
            fold_counts[line.rstrip("\n").split("\t")[fi]] += 1
    print(
        f"legnet TSV fold counts (1=test,2=val,3=train): {dict(sorted(fold_counts.items()))}",
        flush=True,
    )
    print(f"reassign COMPLETED → {OUT_ROOT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
