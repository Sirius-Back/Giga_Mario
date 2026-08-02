"""End-to-end VGAE split assign → ``split.csv`` (pipeline entry)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.splits.vgae.graph_data import load_packed_graph, pack_region_graph
from src.splits.vgae.train import run_vgae_train


def run_vgae_split_assign(
    *,
    outdir: Path,
    graph_dir: Path | None = None,
    pack_dir: Path | None = None,
    marked_dir: Path | None = None,
    seed: int = 42,
    k: int | None = None,
    feature_k: int | None = None,
    feature_ks: tuple[int, ...] | list[int] | None = None,
    per_k_project_dim: int = 256,
    project_dim: int | None = None,
    add_structural_features: bool = False,
    max_ids: int | None = None,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    device: str | None = None,
    homology_table: Path | None = None,
    skip_train: bool = False,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """Pack (if needed) + train VGAE + write ``outdir/split.csv``.

    Returns train meta dict (includes ``split_csv`` path).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_pack = outdir / "pack"

    if pack_dir is not None and (Path(pack_dir) / "feature_meta.json").is_file():
        pack = load_packed_graph(Path(pack_dir))
    else:
        if graph_dir is None or marked_dir is None:
            raise ValueError(
                "run_vgae_split_assign requires graph_dir+marked_dir "
                "or an existing pack_dir with feature_meta.json"
            )
        pack = pack_region_graph(
            Path(graph_dir),
            Path(marked_dir),
            run_pack,
            k=k,
            feature_k=feature_k,
            feature_ks=feature_ks,
            per_k_project_dim=int(per_k_project_dim),
            project_dim=project_dim,
            add_structural_features=bool(add_structural_features),
            max_ids=max_ids,
        )

    if skip_train:
        # Deterministic fallback: size-constrained random on GC only (tests)
        import numpy as np

        from src.splits.vgae.assign import assignment_rows, size_constrained_assign
        from src.splits.sbs.assign import assignment_rows_to_split_csv

        scores = np.zeros((pack.n_nodes, 3), dtype=np.float64)
        scores[:, 0] = pack.x[:, 0]
        scores[:, 1] = -pack.x[:, 0]
        scores[:, 2] = pack.x[:, 0] * 0.1
        labels = size_constrained_assign(scores, ratios=ratios, seed=seed)
        rows = assignment_rows(pack.ids, labels)
        split_csv = assignment_rows_to_split_csv(rows, outdir)
        return {"split_csv": str(split_csv), "skipped_train": True}

    return run_vgae_train(
        pack,
        outdir,
        seed=seed,
        ratios=ratios,
        device=device,
        homology_table=homology_table,
        **train_kwargs,
    )
