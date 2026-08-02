"""Within-model residual leakage: max train similarity and L(τ)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.embed import ROLE_TEST, ROLE_TRAIN, ROLE_VAL
from src.embed.distances import (
    METRICS,
    TrainStats,
    fit_train_stats,
    pairwise_max_similarity,
    prepare_metric_matrix,
)
from src.embed.store import EmbedStore, load_store, mask_role
from src.pipeline.mem_guard import ensure_allocation_fits

DEFAULT_TAU0 = 0.9
DEFAULT_TAU_GRID = np.linspace(0.0, 1.0, 101)


def l_tau(scores: np.ndarray, tau_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """L(τ) = fraction of scores ≥ τ."""
    grid = DEFAULT_TAU_GRID if tau_grid is None else np.asarray(tau_grid, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return grid, np.zeros_like(grid)
    # For each τ: mean(s >= τ)
    # Vectorized via sorting
    s_sorted = np.sort(s)[::-1]
    # For each tau, count
    lt = np.array([(s >= t).mean() for t in grid], dtype=np.float64)
    return grid, lt


def auc_l_tau(tau: np.ndarray, l_vals: np.ndarray) -> float:
    """Trapezoid AUC of L(τ) over τ∈[0,1] (higher → more leakage mass)."""
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(l_vals, tau))


def summarize_scores(scores: np.ndarray, *, tau0: float = DEFAULT_TAU0) -> dict[str, float]:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "mean": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            f"P_ge_{tau0}": float("nan"),
        }
    return {
        "n": int(s.size),
        "median": float(np.median(s)),
        "mean": float(np.mean(s)),
        "p90": float(np.quantile(s, 0.90)),
        "p99": float(np.quantile(s, 0.99)),
        f"P_ge_{tau0}": float((s >= tau0).mean()),
    }


def compute_max_train_sim(
    store: EmbedStore,
    layer: str,
    metric: str,
    *,
    chunk: int = 8192,
    ridge: float = 1e-3,
    include_val: bool = False,
    device: str | None = None,
) -> tuple[np.ndarray, TrainStats, dict[str, Any]]:
    """Return (test_scores, train_stats, meta). Optionally also score val."""
    x = np.asarray(store.layers[layer])
    train_m = mask_role(store.roles, ROLE_TRAIN)
    test_m = mask_role(store.roles, ROLE_TEST)
    train_x = np.asarray(x[train_m], dtype=np.float32)
    test_x = np.asarray(x[test_m], dtype=np.float32)

    # Peak: two transformed matrices
    need = (train_x.nbytes + test_x.nbytes) * 2
    ensure_allocation_fits(need, label=f"leakage_{layer}_{metric}")

    stats = fit_train_stats(train_x, ridge=ridge)
    g = prepare_metric_matrix(train_x, stats, metric).astype(np.float32)
    q = prepare_metric_matrix(test_x, stats, metric).astype(np.float32)
    scores = pairwise_max_similarity(
        q, g, metric=metric, chunk=chunk, device=device
    )

    meta: dict[str, Any] = {
        "layer": layer,
        "metric": metric,
        "n_train": int(train_m.sum()),
        "n_test": int(test_m.sum()),
        "cond": stats.cond,
        "ridge": ridge,
    }
    if include_val:
        val_m = mask_role(store.roles, ROLE_VAL)
        if int(val_m.sum()) > 0:
            v = prepare_metric_matrix(
                np.asarray(x[val_m], dtype=np.float32), stats, metric
            ).astype(np.float32)
            meta["val_scores"] = pairwise_max_similarity(
                v, g, metric=metric, chunk=chunk, device=device
            )
            meta["n_val"] = int(val_m.sum())
    return scores, stats, meta


def run_leakage_for_store(
    store_dir: Path,
    *,
    layers: Iterable[str],
    metrics: Iterable[str] = METRICS,
    out_dir: Path | None = None,
    tau0: float = DEFAULT_TAU0,
    chunk: int = 8192,
    device: str | None = None,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    """Compute L(τ) for all layer×metric; write artifacts under store or out_dir."""
    store_dir = Path(store_dir)
    out = Path(out_dir) if out_dir is not None else store_dir / "leakage"
    out.mkdir(parents=True, exist_ok=True)
    store = load_store(store_dir, layers=layers)
    rows: list[dict[str, Any]] = []

    for layer in layers:
        if layer == "pred":
            # 1-d pred: still valid but whitening/corr degenerate — skip whitened/corr
            use_metrics = [m for m in metrics if m in ("centered_cosine", "l2_euclidean")]
        else:
            use_metrics = list(metrics)
        for metric in use_metrics:
            summary_path = out / f"summary_{layer}_{metric}.json"
            if skip_existing and summary_path.is_file():
                rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            scores, stats, meta = compute_max_train_sim(
                store, layer, metric, chunk=chunk, device=device
            )
            stats.to_npz(out / f"train_stats_{layer}_{metric}.npz")
            np.save(out / f"max_sim_test_{layer}_{metric}.npy", scores.astype(np.float32))
            tau, lt = l_tau(scores)
            np.savez_compressed(
                out / f"L_tau_{layer}_{metric}.npz", tau=tau, L=lt
            )
            summ = summarize_scores(scores, tau0=tau0)
            auc = auc_l_tau(tau, lt)
            row = {
                "store": str(store_dir),
                "layer": layer,
                "metric": metric,
                "auc_L": auc,
                "tau0": tau0,
                **summ,
                "cov_cond": meta.get("cond"),
            }
            rows.append(row)
            (out / f"summary_{layer}_{metric}.json").write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8"
            )
    return rows


def write_ranking_table(rows: list[dict[str, Any]], path: Path) -> Path:
    """TSV ranking: higher median / auc_L → more residual leakage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort within layer×metric by median desc
    header = [
        "run_key",
        "layer",
        "metric",
        "n",
        "median",
        "mean",
        "p90",
        "p99",
        "P_ge_tau0",
        "auc_L",
        "cov_cond",
        "rank_within_layer_metric",
    ]
    # Enrich run_key from store path
    enriched: list[dict[str, Any]] = []
    for r in rows:
        store = Path(r["store"])
        # results/embed_legnet/<run>/[foldK]
        parts = store.parts
        try:
            idx = parts.index("embed_legnet")
            key = "/".join(parts[idx + 1 :])
        except ValueError:
            key = store.name
        er = dict(r)
        er["run_key"] = key
        p_key = [k for k in er if k.startswith("P_ge_")][0]
        er["P_ge_tau0"] = er[p_key]
        enriched.append(er)

    # Rank
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in enriched:
        groups[(r["layer"], r["metric"])].append(r)
    for grp in groups.values():
        grp.sort(key=lambda r: (-float(r["median"]), -float(r["auc_L"])))
        for i, r in enumerate(grp, start=1):
            r["rank_within_layer_metric"] = i

    flat = []
    for grp in groups.values():
        flat.extend(grp)
    flat.sort(key=lambda r: (r["layer"], r["metric"], r["rank_within_layer_metric"]))

    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in flat:
            fh.write(
                "\t".join(
                    str(r.get(h, ""))
                    for h in header
                )
                + "\n"
            )
    return path


def plot_l_tau_curves(
    rows_dir: Path,
    *,
    layer: str,
    metric: str,
    out_pdf: Path,
    run_keys: list[str] | None = None,
) -> Path:
    """Publication-ish L(τ) overlay for one layer×metric across runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Discover L_tau files under results tree
    root = Path(rows_dir)
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for nz in sorted(root.glob(f"**/leakage/L_tau_{layer}_{metric}.npz")):
        key = str(nz.parent.parent.relative_to(root))
        if run_keys is not None and key not in run_keys:
            continue
        z = np.load(nz)
        curves.append((key, z["tau"], z["L"]))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # Okabe–Ito-ish cycle
    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
        "#F0E442",
    ]
    for i, (key, tau, lt) in enumerate(curves):
        ax.plot(
            tau,
            lt,
            label=key.replace("run", "r")[:40],
            color=colors[i % len(colors)],
            lw=1.2,
        )
    ax.set_xlabel(r"similarity threshold $\tau$")
    ax.set_ylabel(r"$L(\tau)=P(s_i \geq \tau)$")
    ax.set_title(f"LegNet residual leakage — {layer} / {metric}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    if curves:
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return out_pdf
