"""Pairwise comparison of LegNet embed stores across split methods.

Compares **geometries** of two representation spaces on a shared ID set:
- RSA / Mantel-style correlation of RDMs built with centered-cosine,
  correlation-distance, and Mahalanobis (whitened Euclidean) distances
- Linear CKA
- Orthogonal Procrustes + matched-vector distances after alignment

Does **not** compute L(τ) leakage curves.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.embed import ROLE_TEST, ROLE_TRAIN
from src.embed.distances import (
    EPS,
    TrainStats,
    fit_train_stats,
    prepare_metric_matrix,
    transform_centered_l2,
    transform_row_centered_l2,
    transform_whitened_l2,
)
from src.embed.store import EmbedStore, load_store, mask_role
from src.pipeline.mem_guard import ensure_allocation_fits

DEFAULT_LAYERS = ("pooled", "stage1_2", "stage0", "head_h")
RDM_METRICS = ("centered_cosine", "corr_distance", "mahalanobis")
HEATMAP_SCORES = (
    "cka_linear",
    "rsa_centered_cosine",
    "rsa_corr_distance",
    "rsa_mahalanobis",
    "procrustes_disparity",
    "mean_centered_cosine_dist",
    "mean_mahalanobis",
)

_RUN_PREFIX_RE = re.compile(r"^run\d+_(?:legnet|caduceus)_")
_R_PREFIX_RE = re.compile(r"^r\d+_")
_FOLD_SUFFIX_RE = re.compile(r"/fold\d+$")


def short_run_label(key: str) -> str:
    """Drop run prefixes / LOO fold; underscores → spaces; uppercase.

    ``pangenome`` is abbreviated to ``PG``.
    """
    s = str(key)
    s = _RUN_PREFIX_RE.sub("", s)
    s = _R_PREFIX_RE.sub("", s)
    s = _FOLD_SUFFIX_RE.sub("", s)
    s = re.sub(r"pangenome", "PG", s, flags=re.IGNORECASE)
    s = s.replace("_", " ").strip()
    return s.upper()


def _upper_tri(dist: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(dist, k=1)
    return dist[idx]


def pairwise_distance_matrix(
    x: np.ndarray, *, metric: str, chunk: int = 1024
) -> np.ndarray:
    """Return condensed-friendly square distance matrix for unit / whitened rows.

    ``x`` must already be transformed (L2 rows for cosine-family; whitened for
    Mahalanobis — then Euclidean on whitened = Mahalanobis in original).
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    if metric in ("centered_cosine", "corr_distance"):
        # cosine distance = 1 - cos; x rows unit-norm
        for i in range(0, n, chunk):
            sl = x[i : i + chunk]
            sims = sl @ x.T
            np.clip(sims, -1.0, 1.0, out=sims)
            out[i : i + len(sl)] = 1.0 - sims
        np.fill_diagonal(out, 0.0)
        return out
    if metric == "mahalanobis":
        # Euclidean on already-whitened rows
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        nrm = np.sum(x * x, axis=1)
        for i in range(0, n, chunk):
            sl = x[i : i + chunk]
            dots = sl @ x.T
            d2 = nrm[i : i + len(sl), None] + nrm[None, :] - 2.0 * dots
            out[i : i + len(sl)] = np.sqrt(np.maximum(d2, 0.0))
        np.fill_diagonal(out, 0.0)
        return out
    raise ValueError(f"unknown RDM metric {metric!r}")


def transform_for_rdm(
    x: np.ndarray, stats: TrainStats, metric: str
) -> np.ndarray:
    if metric == "centered_cosine":
        return transform_centered_l2(x, stats).astype(np.float32)
    if metric == "corr_distance":
        return transform_row_centered_l2(x).astype(np.float32)
    if metric == "mahalanobis":
        return transform_whitened_l2(x, stats).astype(np.float32)
    raise ValueError(metric)


def rsa_spearman(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman correlation of upper triangles (RSA)."""
    a = _upper_tri(rdm_a)
    b = _upper_tri(rdm_b)
    if a.size < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(a, b)
        return float(rho)
    except ImportError:
        # Rank-transform + Pearson
        def _rank(v: np.ndarray) -> np.ndarray:
            order = np.argsort(v, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(v) + 1, dtype=np.float64)
            return ranks

        ra, rb = _rank(a), _rank(b)
        ra -= ra.mean()
        rb -= rb.mean()
        return float(np.sum(ra * rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + EPS))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between ``[N, Dx]`` and ``[N, Dy]`` (rows = same objects)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xtx = x.T @ x
    yty = y.T @ y
    xty = x.T @ y
    hsic_xy = float(np.sum(xty**2))
    hsic_xx = float(np.sum(xtx**2))
    hsic_yy = float(np.sum(yty**2))
    denom = np.sqrt(hsic_xx * hsic_yy) + EPS
    return hsic_xy / denom


def orthogonal_procrustes(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float]:
    """Align ``y`` to ``x`` with orthogonal R (same N, possibly different D).

    Pads the smaller feature dim with zeros. Returns ``(y_aligned, disparity)``
    where disparity = ||X - Y_aligned||_F^2 / ||X||_F^2.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    n, dx = x.shape
    dy = y.shape[1]
    d = max(dx, dy)
    xp = np.zeros((n, d), dtype=np.float64)
    yp = np.zeros((n, d), dtype=np.float64)
    xp[:, :dx] = x
    yp[:, :dy] = y
    # Solve min ||X - Y R|| ; R = UV^T from SVD of Y^T X
    u, _, vt = np.linalg.svd(yp.T @ xp, full_matrices=False)
    r = u @ vt
    y_al = yp @ r
    num = float(np.sum((xp - y_al) ** 2))
    den = float(np.sum(xp**2)) + EPS
    return y_al[:, :dx], num / den


def matched_mean_distances(
    x: np.ndarray, y_aligned: np.ndarray, stats_x: TrainStats
) -> dict[str, float]:
    """Mean distance between matched rows after Procrustes (in X's geometry)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_aligned, dtype=np.float64)
    # Centered cosine distance
    xc = transform_centered_l2(x, stats_x)
    # y already in X coords; use same mean from stats_x
    yc = transform_centered_l2(y, stats_x)
    cos = np.sum(xc * yc, axis=1)
    cos = np.clip(cos, -1.0, 1.0)
    d_cos = 1.0 - cos
    # Correlation distance (row-center then cosine)
    xr = transform_row_centered_l2(x)
    yr = transform_row_centered_l2(y)
    cos_r = np.clip(np.sum(xr * yr, axis=1), -1.0, 1.0)
    d_corr = 1.0 - cos_r
    # Mahalanobis = Euclidean after X whitening
    xw = transform_whitened_l2(x, stats_x)
    yw = transform_whitened_l2(y, stats_x)
    d_mah = np.linalg.norm(xw - yw, axis=1)
    return {
        "mean_centered_cosine_dist": float(np.mean(d_cos)),
        "mean_corr_dist": float(np.mean(d_corr)),
        "mean_mahalanobis": float(np.mean(d_mah)),
        "median_centered_cosine_dist": float(np.median(d_cos)),
        "median_corr_dist": float(np.median(d_corr)),
        "median_mahalanobis": float(np.median(d_mah)),
    }


_STORE_FOLD_RE = re.compile(r"/fold(\d+)$")


def discover_stores(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p.parent
        for p in root.glob("**/manifest.json")
        if p.parent.name != "leakage" and p.parent.name != "pairwise"
    )


def store_key(store_dir: Path, root: Path) -> str:
    try:
        return str(store_dir.relative_to(root))
    except ValueError:
        return store_dir.name


def filter_loo_store_keys(
    keys: list[str], loo_fold: int | None
) -> list[str]:
    """Keep only ``…/fold{loo_fold}`` for LOO stores; pass non-LOO through.

    ``loo_fold=None`` keeps every fold. Publication default is ``0``.
    """
    if loo_fold is None:
        return list(keys)
    out: list[str] = []
    for k in keys:
        m = _STORE_FOLD_RE.search(k)
        if m is None or int(m.group(1)) == loo_fold:
            out.append(k)
    return out


def plot_lower_triangle_hypotenuse(
    mat: np.ndarray,
    labels: list[str],
    *,
    title: str,
    out_pdf: Path,
    out_svg: Path,
    cmap: str,
    label_fontsize: float = 20.0,
) -> None:
    """Strict lower-triangle heatmap rotated with open hypotenuse at the bottom.

    Matrix coords ``(col=j, row=i)`` are rotated −45°:
    ``x'=(j+i)/√2``, ``y'=(-j+i)/√2``. The diagonal (constant self-scores) is
    omitted so the bottom edge is open.

    Labels are placed on the **two legs only**, baselines perpendicular to each
    cathetus.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mat = np.asarray(mat, dtype=np.float64)
    n = mat.shape[0]
    if mat.shape != (n, n) or len(labels) != n:
        raise ValueError("mat must be square and match labels length")

    # Cell corners in (x=col, y=row); strict lower triangle (no diagonal)
    xs = np.arange(n + 1, dtype=np.float64)
    ys = np.arange(n + 1, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    sqrt2 = np.sqrt(2.0)
    Xr = (X + Y) * inv_sqrt2
    Yr = (-X + Y) * inv_sqrt2

    # Mask upper triangle AND diagonal (self-comparisons are constant)
    C = np.ma.array(mat, mask=np.triu(np.ones((n, n), dtype=bool), k=0))
    finite = C.compressed()
    if finite.size == 0:
        raise RuntimeError(f"no finite values for {title}")
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    max_lab = max((len(s) for s in labels), default=1)
    # Outward offset from each leg; grow with label length / font
    pad = 0.35 + 0.018 * max_lab + 0.015 * label_fontsize

    width = max(14.0, 0.95 * n + 7.0)
    height = max(9.0, 0.6 * n + 5.5)
    fig, ax = plt.subplots(figsize=(width, height))
    mesh = ax.pcolormesh(
        Xr,
        Yr,
        C,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=max(14.0, label_fontsize + 2), pad=14)

    # Unit outward normals after −45° rotation:
    # left edge direction (1,1) @45° → outward (−1,1); text ⊥ edge → rotation=-45°
    # right edge direction (1,−1) @-45° → outward (1,1); text ⊥ edge → rotation=+45°
    n_left = np.array([-inv_sqrt2, inv_sqrt2])
    n_right = np.array([inv_sqrt2, inv_sqrt2])

    for k, lab in enumerate(labels):
        # --- left leg (column j=0, row k) ---
        lx = (k + 0.5) * inv_sqrt2
        ly = (k + 0.5) * inv_sqrt2
        ax.text(
            lx + n_left[0] * pad,
            ly + n_left[1] * pad,
            lab,
            ha="right",
            va="center",
            rotation=-45,
            rotation_mode="anchor",
            fontsize=label_fontsize,
            clip_on=False,
        )

        # --- right leg (row i=n edge, column k) ---
        rx = (n + k + 0.5) * inv_sqrt2
        ry = (n - k - 0.5) * inv_sqrt2
        ax.text(
            rx + n_right[0] * pad,
            ry + n_right[1] * pad,
            lab,
            ha="left",
            va="center",
            rotation=45,
            rotation_mode="anchor",
            fontsize=label_fontsize,
            clip_on=False,
        )

    # Margins for leg labels (no hypotenuse text)
    x_max = n * sqrt2
    y_max = n * inv_sqrt2
    margin = pad + 0.7 + 0.05 * max_lab
    ax.set_xlim(-margin, x_max + margin)
    ax.set_ylim(-0.35, y_max + margin)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.0%", pad=0.65)
    cb = fig.colorbar(mesh, cax=cax)
    cb.ax.tick_params(labelsize=max(11.0, label_fontsize - 1))
    cb.outline.set_linewidth(1.0)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def write_heatmaps_from_matrices(
    out_dir: Path,
    keys: list[str],
    *,
    layers: Iterable[str],
    scores: Iterable[str] = HEATMAP_SCORES,
    label_fontsize: float = 20.0,
) -> list[Path]:
    """Rebuild publication triangle heatmaps from ``matrix_{layer}_{score}.npy``."""
    out_dir = Path(out_dir)
    labels = [short_run_label(k) for k in keys]
    written: list[Path] = []
    for layer in layers:
        for score in scores:
            path = out_dir / f"matrix_{layer}_{score}.npy"
            if not path.is_file():
                print(f"[replot] skip missing {path}", flush=True)
                continue
            mat = np.load(path)
            cmap = (
                "viridis"
                if ("rsa" in score or score == "cka_linear")
                else "magma"
            )
            pdf = out_dir / f"heatmap_{layer}_{score}.pdf"
            svg = out_dir / f"heatmap_{layer}_{score}.svg"
            plot_lower_triangle_hypotenuse(
                mat,
                labels,
                title=f"{score} — {layer}",
                out_pdf=pdf,
                out_svg=svg,
                cmap=cmap,
                label_fontsize=label_fontsize,
            )
            written.extend([pdf, svg])
            print(f"[replot] {pdf.name}", flush=True)
    return written


def _id_index(st: EmbedStore) -> dict[str, int]:
    return {str(i): j for j, i in enumerate(st.ids)}


def _fit_store_stats(
    st: EmbedStore, layer: str, *, seed: int, max_train: int = 50000
) -> TrainStats:
    rng = np.random.default_rng(seed)
    tr = mask_role(st.roles, ROLE_TRAIN)
    train_x = np.asarray(st.layers[layer][tr], dtype=np.float32)
    if train_x.shape[0] > max_train:
        idx = rng.choice(train_x.shape[0], size=max_train, replace=False)
        train_x = train_x[idx]
    return fit_train_stats(train_x, ridge=1e-3)


def _pair_aligned(
    st_a: EmbedStore,
    st_b: EmbedStore,
    layer: str,
    *,
    role: str,
    max_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two stores on a shared ID set.

    ``role``:
      - ``all`` — any ID present in both stores (recommended across splits)
      - ``test`` — IDs that are test in **both** (often small / empty)
      - ``test_either`` — test in at least one store, present in both
    """
    ia, ib = _id_index(st_a), _id_index(st_b)
    ids_a, ids_b = set(ia), set(ib)
    common = ids_a & ids_b
    if role == "test":
        ta = {
            str(i)
            for i, r in zip(st_a.ids, st_a.roles)
            if int(r) == ROLE_TEST
        }
        tb = {
            str(i)
            for i, r in zip(st_b.ids, st_b.roles)
            if int(r) == ROLE_TEST
        }
        common &= ta & tb
    elif role == "test_either":
        ta = {
            str(i)
            for i, r in zip(st_a.ids, st_a.roles)
            if int(r) == ROLE_TEST
        }
        tb = {
            str(i)
            for i, r in zip(st_b.ids, st_b.roles)
            if int(r) == ROLE_TEST
        }
        common &= ta | tb
    elif role != "all":
        raise ValueError(f"unknown role mode {role!r}")

    if len(common) < 32:
        raise RuntimeError(
            f"pair ID intersection too small: n={len(common)} (role={role})"
        )
    common_sorted = sorted(common, key=lambda x: (len(x), x))
    rng = np.random.default_rng(seed)
    if len(common_sorted) > max_n:
        pick = rng.choice(len(common_sorted), size=max_n, replace=False)
        pick.sort()
        common_sorted = [common_sorted[i] for i in pick]
    xa = np.asarray(
        st_a.layers[layer][[ia[i] for i in common_sorted]], dtype=np.float32
    )
    xb = np.asarray(
        st_b.layers[layer][[ib[i] for i in common_sorted]], dtype=np.float32
    )
    return xa, xb, common_sorted


def compare_pair(
    key_a: str,
    key_b: str,
    xa: np.ndarray,
    xb: np.ndarray,
    stats_a: TrainStats,
    stats_b: TrainStats,
    *,
    rdm_n: int,
    seed: int,
) -> dict[str, Any]:
    """One pair on one layer; xa/xb already ID-aligned."""
    n = xa.shape[0]
    rng = np.random.default_rng(seed)
    if n > rdm_n:
        idx = rng.choice(n, size=rdm_n, replace=False)
        idx.sort()
        xa_r, xb_r = xa[idx], xb[idx]
    else:
        xa_r, xb_r = xa, xb

    row: dict[str, Any] = {
        "run_a": key_a,
        "run_b": key_b,
        "n_aligned": int(n),
        "n_rdm": int(xa_r.shape[0]),
    }

    # RSA for each distance metric
    for metric in RDM_METRICS:
        ta = transform_for_rdm(xa_r, stats_a, metric)
        tb = transform_for_rdm(xb_r, stats_b, metric)
        need = 2 * (ta.nbytes + int(ta.shape[0]) ** 2 * 4)
        ensure_allocation_fits(need, label=f"rdm_{metric}")
        da = pairwise_distance_matrix(ta, metric=metric)
        db = pairwise_distance_matrix(tb, metric=metric)
        row[f"rsa_{metric}"] = rsa_spearman(da, db)

    # CKA on full aligned set (centered inside)
    row["cka_linear"] = linear_cka(xa, xb)

    # Procrustes: align B → A, then matched distances in A's geometry
    y_al, disp = orthogonal_procrustes(xa, xb)
    row["procrustes_disparity"] = float(disp)
    matched = matched_mean_distances(xa, y_al, stats_a)
    row.update(matched)
    return row


def run_pairwise_compare(
    embed_root: Path,
    out_dir: Path,
    *,
    layers: Iterable[str] = DEFAULT_LAYERS,
    role: str = "all",
    max_n: int = 8192,
    rdm_n: int = 2048,
    seed: int = 42,
    loo_fold: int | None = 0,
) -> Path:
    """Compare all store pairs; write TSV + JSON + heatmaps.

    ``role`` defaults to ``all`` because different splits have disjoint test
    sets — a global test∩test intersection is empty.

    ``loo_fold`` (default 0) keeps a single LOO fold per LOO run so heatmaps
    are not dominated by repeated r31 fold axes; ``None`` keeps all folds.
    """
    embed_root = Path(embed_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if role not in {"all", "test", "test_either"}:
        raise ValueError("role must be all|test|test_either")

    store_dirs = discover_stores(embed_root)
    if len(store_dirs) < 2:
        raise RuntimeError(f"need ≥2 stores under {embed_root}")

    all_keys = [store_key(d, embed_root) for d in store_dirs]
    keys = filter_loo_store_keys(sorted(all_keys), loo_fold)
    keep = set(keys)
    store_dirs = [d for d in store_dirs if store_key(d, embed_root) in keep]
    stores = {
        store_key(d, embed_root): load_store(d, layers=layers) for d in store_dirs
    }
    keys = sorted(stores)
    if len(keys) < 2:
        raise RuntimeError(
            f"need ≥2 stores after loo_fold={loo_fold!r} under {embed_root}"
        )
    all_rows: list[dict[str, Any]] = []

    # Precompute train stats per (run, layer)
    stats_cache: dict[tuple[str, str], TrainStats] = {}

    for layer in layers:
        print(f"[pairwise] layer={layer} n_runs={len(keys)}", flush=True)
        for k in keys:
            stats_cache[(k, layer)] = _fit_store_stats(
                stores[k], layer, seed=seed + hash((k, layer)) % 100000
            )
        layer_rows: list[dict[str, Any]] = []
        for a, b in combinations(keys, 2):
            try:
                xa, xb, ids = _pair_aligned(
                    stores[a],
                    stores[b],
                    layer,
                    role=role,
                    max_n=max_n,
                    seed=seed + hash((a, b, layer)) % 100000,
                )
            except RuntimeError as exc:
                print(f"  SKIP {a} vs {b}: {exc}", flush=True)
                continue
            row = compare_pair(
                a,
                b,
                xa,
                xb,
                stats_cache[(a, layer)],
                stats_cache[(b, layer)],
                rdm_n=rdm_n,
                seed=seed + hash((a, b, layer, "cmp")) % 100000,
            )
            row["layer"] = layer
            row["id_role"] = role
            layer_rows.append(row)
            all_rows.append(row)
            print(
                f"  {a} vs {b}: n={len(ids)} cka={row['cka_linear']:.3f} "
                f"rsa_cos={row['rsa_centered_cosine']:.3f} "
                f"proc={row['procrustes_disparity']:.3f}",
                flush=True,
            )

        # Heatmaps for key scores (lower triangle on hypotenuse; short labels)
        labels = [short_run_label(k) for k in keys]
        for score in HEATMAP_SCORES:
            mat = np.full((len(keys), len(keys)), np.nan, dtype=np.float64)
            for i, ki in enumerate(keys):
                mat[i, i] = 1.0 if not score.startswith(("procrustes", "mean_")) else 0.0
            for r in layer_rows:
                i, j = keys.index(r["run_a"]), keys.index(r["run_b"])
                mat[i, j] = mat[j, i] = r[score]
            np.save(out_dir / f"matrix_{layer}_{score}.npy", mat)
            cmap = (
                "viridis"
                if ("rsa" in score or score == "cka_linear")
                else "magma"
            )
            plot_lower_triangle_hypotenuse(
                mat,
                labels,
                title=f"{score} — {layer}",
                out_pdf=out_dir / f"heatmap_{layer}_{score}.pdf",
                out_svg=out_dir / f"heatmap_{layer}_{score}.svg",
                cmap=cmap,
            )

    # Write TSV
    tsv = out_dir / "pairwise_compare.tsv"
    if all_rows:
        cols = list(all_rows[0].keys())
        with tsv.open("w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in all_rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    (out_dir / "pairwise_compare.json").write_text(
        json.dumps(
            {
                "embed_root": str(embed_root),
                "role": role,
                "max_n": max_n,
                "rdm_n": rdm_n,
                "seed": seed,
                "loo_fold": loo_fold,
                "layers": list(layers),
                "runs": keys,
                "n_pairs": len(all_rows),
                "rows": all_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[pairwise] wrote {tsv}", flush=True)
    return tsv
