"""Feature table → SBS assignment table + split.csv glue.

Clustering operates on the feature matrix (O(n·d)), not a dense distance matrix.
Default method: DBSCAN. Also: kmeans, kmeans_elbow, hierarchical, pca_kmeans.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from src.pipeline.common import write_csv
from src.pipeline.generate_fold import is_zsv_fold, normalize_fold_label
from src.splits.common import assign_folds_random, assign_folds_stratified
from src.splits.sbs.features import FeatureTable

ASSIGNMENT_COLUMNS = ["region", "cluster", "train_test", "fold", "additional"]

CLUSTER_METHODS: tuple[str, ...] = (
    "dbscan",
    "kmeans",
    "kmeans_elbow",
    "hierarchical",
    "pca_kmeans",
    "auto",
)

ClusterMethod = Literal[
    "dbscan",
    "kmeans",
    "kmeans_elbow",
    "hierarchical",
    "pca_kmeans",
    "auto",
]

# Aliases accepted from CLI / Hydra / split captions.
_CLUSTER_METHOD_ALIASES: dict[str, ClusterMethod] = {
    "dbscan": "dbscan",
    "dbs": "dbscan",
    "kmeans": "kmeans",
    "k-means": "kmeans",
    "km": "kmeans",
    "kmeans_elbow": "kmeans_elbow",
    "elbow": "kmeans_elbow",
    "kmeans-elbow": "kmeans_elbow",
    "hierarchical": "hierarchical",
    "hclust": "hierarchical",
    "agglomerative": "hierarchical",
    "pca_kmeans": "pca_kmeans",
    "pca-kmeans": "pca_kmeans",
    "pca+kmeans": "pca_kmeans",
    "auto": "auto",
}


def normalize_cluster_method(method: str | ClusterMethod) -> ClusterMethod:
    """Map user / config aliases onto a supported ClusterMethod."""
    key = str(method).strip().lower().replace(" ", "_")
    if key not in _CLUSTER_METHOD_ALIASES:
        raise ValueError(
            f"Unknown cluster_method={method!r}; "
            f"supported: {', '.join(CLUSTER_METHODS)}"
        )
    return _CLUSTER_METHOD_ALIASES[key]


def _standardize(x: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """Column z-score.

    For large float32 matrices, ``inplace=True`` mutates ``x`` and returns it
    (avoids a second ~n×d resident copy that OOM-killed full-panel k=7 assign).
    """
    x = np.asarray(x)
    if x.dtype == np.float32:
        out = x if inplace else np.array(x, dtype=np.float32, copy=True, order="C")
        mu = out.mean(axis=0)
        sd = out.std(axis=0)
        sd = np.where(sd < 1e-12, np.float32(1.0), sd.astype(np.float32, copy=False))
        out -= mu
        out /= sd
        return out
    if inplace:
        raise ValueError("inplace standardize only supported for float32 matrices")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """Dense pairwise distances — only for small-n hierarchical/silhouette."""
    n = x.shape[0]
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        diff = x[i] - x
        out[i] = np.sqrt(np.sum(diff * diff, axis=1))
    return out


def _silhouette_features(x: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    uniq = set(int(v) for v in labels.tolist())
    # DBSCAN noise (-1) / single cluster → undefined
    valid = [u for u in uniq if u >= 0]
    if len(valid) < 2:
        return float("nan")
    mask = labels >= 0
    if int(mask.sum()) < 3:
        return float("nan")
    try:
        return float(silhouette_score(x[mask], labels[mask], metric="euclidean"))
    except Exception:  # noqa: BLE001
        return float("nan")


def _elbow_k(inertias: dict[int, float]) -> int:
    """Pick k maximizing distance from the first–last inertia chord."""
    ks = sorted(inertias)
    if len(ks) == 1:
        return ks[0]
    x = np.asarray(ks, dtype=float)
    y = np.asarray([inertias[k] for k in ks], dtype=float)
    # Normalize to unit box
    x_n = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    line = p2 - p1
    norm = np.linalg.norm(line)
    if norm < 1e-12:
        return ks[0]
    dists = []
    for i in range(len(ks)):
        p = np.array([x_n[i], y_n[i]])
        # 2D cross magnitude / |line| = perpendicular distance to chord
        cross = (p2[0] - p1[0]) * (p1[1] - p[1]) - (p1[0] - p[0]) * (p2[1] - p[1])
        dists.append(abs(cross) / norm)
    return ks[int(np.argmax(dists))]


# DBSCAN neighbor graphs become memory-hostile well below full-panel MARKED sizes.
DBSCAN_MAX_N = 50_000
# Prefer MiniBatchKMeans above this for kmeans / elbow paths.
MINIBATCH_KMEANS_N = 20_000


def _cluster_dbscan(
    x: np.ndarray, *, eps: float | None, min_samples: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    n = x.shape[0]
    if n > DBSCAN_MAX_N:
        raise ValueError(
            f"DBSCAN refused for n={n} (>{DBSCAN_MAX_N}); "
            "use cluster_method=kmeans_elbow|kmeans|pca_kmeans|auto"
        )
    meta: dict[str, Any] = {"min_samples": min_samples}
    if eps is None:
        # k-distance heuristic: median distance to min_samples-th neighbor
        k = min(max(min_samples, 2), max(n - 1, 1))
        nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
        nn.fit(x)
        dists, _ = nn.kneighbors(x)
        eps = float(np.median(dists[:, -1]))
        if eps <= 0:
            eps = 0.5
        meta["eps_auto"] = eps
    else:
        meta["eps"] = eps
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree")
    labels = model.fit_predict(x)
    # Remap noise (-1) to its own singleton-style cluster ids after core labels
    remapped = labels.copy()
    next_id = int(remapped[remapped >= 0].max()) + 1 if np.any(remapped >= 0) else 0
    for i, lab in enumerate(remapped):
        if lab < 0:
            remapped[i] = next_id
            next_id += 1
    meta["n_noise_as_singletons"] = int(np.sum(labels < 0))
    meta["n_core_clusters"] = int(len(set(labels.tolist()) - {-1}))
    _ = seed  # DBSCAN is deterministic given data; seed kept for API parity
    return remapped, meta


def _cluster_kmeans(x: np.ndarray, n_clusters: int, *, seed: int) -> np.ndarray:
    n = x.shape[0]
    k = min(max(int(n_clusters), 1), n)
    if k >= n:
        return np.arange(n, dtype=int)
    if n >= MINIBATCH_KMEANS_N:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(4096, max(256, n // 50)),
            n_init=10,
        )
    else:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return model.fit_predict(x)


def _cluster_kmeans_elbow(
    x: np.ndarray,
    *,
    seed: int,
    k_min: int,
    k_max: int | None,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = x.shape[0]
    # Full-panel elbow: cap k search — many MiniBatch fits × n×d blew RAM (run13/14).
    hard_cap = 8 if n >= 100_000 else 20
    upper = min(k_max or max(3, int(np.sqrt(n))), n - 1 if n > 1 else 1, hard_cap)
    lower = max(1, min(k_min, upper))
    inertias: dict[int, float] = {}
    if checkpoint_path is not None and checkpoint_path.is_file():
        try:
            raw = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for kk, vv in (raw.get("inertias") or {}).items():
                inertias[int(kk)] = float(vv)
            print(
                f"[assign] resumed elbow inertias from {checkpoint_path} "
                f"({len(inertias)} ks)",
                flush=True,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[assign] ignoring bad elbow checkpoint: {exc}", flush=True)
            inertias = {}
    n_init = 3 if n >= 100_000 else (5 if n >= MINIBATCH_KMEANS_N else 10)
    for k in range(lower, upper + 1):
        if k in inertias:
            print(f"[assign] kmeans_elbow skip k={k} (checkpointed)", flush=True)
            continue
        if n >= MINIBATCH_KMEANS_N:
            from sklearn.cluster import MiniBatchKMeans

            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                batch_size=min(4096, max(256, n // 50)),
                n_init=n_init,
            )
        else:
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        print(f"[assign] kmeans_elbow fit k={k}/{upper} n={n}", flush=True)
        model.fit(x)
        inertias[k] = float(model.inertia_)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "inertias": {str(a): b for a, b in sorted(inertias.items())},
                        "lower": lower,
                        "upper": upper,
                        "n": n,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            tmp.replace(checkpoint_path)
    k_best = _elbow_k(inertias)
    print(f"[assign] kmeans_elbow best k={k_best}; final fit", flush=True)
    labels = _cluster_kmeans(x, k_best, seed=seed)
    return labels, {
        "k": k_best,
        "inertias": inertias,
        "selection": "elbow",
        "k_max_cap": hard_cap,
        "n_init": n_init,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


def _cluster_hierarchical(x: np.ndarray, n_clusters: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = x.shape[0]
    k = min(max(int(n_clusters), 1), n)
    if k >= n:
        return np.arange(n, dtype=int)
    if n > 5000:
        raise ValueError(
            f"hierarchical clustering on features refused for n={n} (>5000); "
            "use dbscan or kmeans"
        )
    condensed = pdist(x, metric="euclidean")
    z = linkage(condensed, method="average")
    return fcluster(z, t=k, criterion="maxclust") - 1


def _cluster_pca_kmeans(x: np.ndarray, n_clusters: int, *, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    n = x.shape[0]
    k = min(max(int(n_clusters), 1), n)
    if k >= n:
        return np.arange(n, dtype=int)
    n_comp = min(max(k, 2), max(n - 1, 1), x.shape[1], 16)
    emb = PCA(n_components=n_comp, random_state=seed).fit_transform(x)
    return _cluster_kmeans(emb, k, seed=seed)


def choose_n_clusters_features(
    x: np.ndarray,
    *,
    method: ClusterMethod,
    seed: int,
    k_min: int = 2,
    k_max: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Silhouette-based k for methods that need an explicit cluster count."""
    n = x.shape[0]
    if n < 3:
        return 1, {"reason": "n<3", "scores": {}}
    upper = min(k_max or max(3, int(np.sqrt(n))), n - 1, 12)
    lower = max(2, min(k_min, upper))
    methods: list[str]
    if method == "auto":
        methods = ["kmeans", "pca_kmeans"]
    elif method in ("kmeans", "pca_kmeans", "hierarchical"):
        methods = [method]
    else:
        return lower, {"reason": f"{method} does not use choose_n_clusters", "scores": {}}

    best_k = lower
    best_score = float("-inf")
    best_method = methods[0]
    scores: dict[str, dict[int, float]] = {}
    for meth in methods:
        scores[meth] = {}
        for k in range(lower, upper + 1):
            if meth == "hierarchical":
                labels = _cluster_hierarchical(x, k)
            elif meth == "pca_kmeans":
                labels = _cluster_pca_kmeans(x, k, seed=seed)
            else:
                labels = _cluster_kmeans(x, k, seed=seed)
            score = _silhouette_features(x, labels)
            scores[meth][k] = score
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_k = k
                best_method = meth
    return best_k, {
        "best_method": best_method,
        "best_score": best_score if np.isfinite(best_score) else None,
        "scores": scores,
    }


def cluster_feature_table(
    features: FeatureTable,
    *,
    n_clusters: int | Literal["auto"] = "auto",
    method: str | ClusterMethod = "dbscan",
    seed: int = 42,
    k_min: int = 2,
    k_max: int | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    elbow_checkpoint_path: Path | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Return region→cluster id map from the feature matrix.

    ``method`` selects the clustering backend (aliases accepted via
    :func:`normalize_cluster_method`). Large-n guards:

    - ``dbscan`` / ``auto`` with n > ``DBSCAN_MAX_N`` → MiniBatchKMeans elbow
    - ``hierarchical`` with n > 5000 → ValueError (caller should switch method)
    - float32 panels with n×d ≥ 5e6: standardize **in place** (no second matrix copy)
    """
    method = normalize_cluster_method(method)
    n_cells = int(features.n) * int(features.n_features)
    inplace = features.matrix.dtype == np.float32 and n_cells >= 5_000_000
    x = _standardize(features.matrix, inplace=inplace)
    meta: dict[str, Any] = {
        "method_requested": method,
        "feature_names": list(features.feature_names),
        "supported_methods": list(CLUSTER_METHODS),
        "standardize_inplace": inplace,
    }
    used: str
    labels: np.ndarray
    k_info: dict[str, Any] = {}
    n = int(x.shape[0])

    def _elbow() -> tuple[np.ndarray, dict[str, Any]]:
        return _cluster_kmeans_elbow(
            x,
            seed=seed,
            k_min=k_min,
            k_max=k_max,
            checkpoint_path=elbow_checkpoint_path,
        )

    # --- dispatch by method (with documented large-n fallbacks) ---
    if method in ("dbscan", "auto") and n > DBSCAN_MAX_N:
        labels, k_info = _elbow()
        used = "kmeans_elbow"
        meta["fallback"] = "dbscan_large_n"
        meta["dbscan_max_n"] = DBSCAN_MAX_N
    elif method == "auto":
        labels, db_meta = _cluster_dbscan(
            x, eps=dbscan_eps, min_samples=dbscan_min_samples, seed=seed
        )
        meta["dbscan"] = db_meta
        n_lab = len(set(int(v) for v in labels.tolist()))
        if n_lab < 2:
            labels, k_info = _elbow()
            used = "kmeans_elbow"
            meta["fallback"] = "dbscan_collapsed"
        else:
            used = "dbscan"
    elif method == "dbscan":
        labels, db_meta = _cluster_dbscan(
            x, eps=dbscan_eps, min_samples=dbscan_min_samples, seed=seed
        )
        meta["dbscan"] = db_meta
        used = "dbscan"
    elif method == "kmeans_elbow":
        labels, k_info = _elbow()
        used = "kmeans_elbow"
    elif method == "hierarchical":
        if n_clusters == "auto":
            k, choice = choose_n_clusters_features(
                x, method="hierarchical", seed=seed, k_min=k_min, k_max=k_max
            )
            meta["k_selection"] = choice
        else:
            k = int(n_clusters)
        labels = _cluster_hierarchical(x, k)
        used = "hierarchical"
        k_info = {"k": k}
    elif method == "pca_kmeans":
        if n_clusters == "auto":
            k, choice = choose_n_clusters_features(
                x, method="pca_kmeans", seed=seed, k_min=k_min, k_max=k_max
            )
            meta["k_selection"] = choice
        else:
            k = int(n_clusters)
        labels = _cluster_pca_kmeans(x, k, seed=seed)
        used = "pca_kmeans"
        k_info = {"k": k}
    elif method == "kmeans":
        if n_clusters == "auto":
            k, choice = choose_n_clusters_features(
                x, method="kmeans", seed=seed, k_min=k_min, k_max=k_max
            )
            meta["k_selection"] = choice
        else:
            k = int(n_clusters)
        labels = _cluster_kmeans(x, k, seed=seed)
        used = "kmeans"
        k_info = {"k": k}
    else:  # pragma: no cover — normalize_cluster_method guards this
        raise ValueError(f"Unhandled cluster_method={method!r}")

    meta["method_used"] = used
    meta["n_clusters"] = int(len(set(int(v) for v in labels.tolist())))
    if k_info:
        meta["k_info"] = k_info
    mapping = {rid: int(lab) for rid, lab in zip(features.ids, labels)}
    return mapping, meta


def _load_id_fold_map(fold_csv: Path | None) -> dict[str, str]:
    if fold_csv is None:
        return {}
    from src.pipeline.common import read_csv

    rows = read_csv(Path(fold_csv))
    out: dict[str, str] = {}
    for row in rows:
        rid = row["ID"].strip()
        out[rid] = normalize_fold_label(row["fold"])
    return out


def _load_strat_map(strat_csv: Path | None) -> dict[str, dict[str, str]]:
    if strat_csv is None:
        return {}
    from src.pipeline.common import read_csv

    rows = read_csv(Path(strat_csv))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        rid = row["ID"].strip()
        out[rid] = row
    return out


def _strat_columns(sample: dict[str, str]) -> list[str]:
    skip = {"ID", "id"}
    cols = [c for c in sample if c not in skip]
    strat_star = [c for c in cols if c.lower().startswith("strat")]
    return strat_star if strat_star else cols


def _is_numeric(values: Sequence[str]) -> bool:
    if not values:
        return False
    for v in values:
        if v is None or str(v).strip() == "":
            continue
        try:
            float(v)
        except ValueError:
            return False
    return True


def aggregate_stratification_per_fold(
    fold_members: dict[str, list[str]],
    strat_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Aggregate stratification.csv onto folds.

    Numeric columns → sum; categorical → mode. Composite key joins all columns.
    """
    if not strat_map:
        return {fold: "all" for fold in fold_members}
    sample = next(iter(strat_map.values()))
    cols = _strat_columns(sample)
    if not cols:
        return {fold: "all" for fold in fold_members}

    col_numeric = {
        c: _is_numeric([row.get(c, "") for row in strat_map.values()]) for c in cols
    }
    fold_strata: dict[str, str] = {}
    for fold, members in fold_members.items():
        parts: list[str] = []
        for col in cols:
            vals = [
                strat_map[m][col]
                for m in members
                if m in strat_map and col in strat_map[m]
            ]
            if not vals:
                parts.append(f"{col}=NA")
                continue
            if col_numeric[col]:
                total = sum(float(v) for v in vals if str(v).strip() != "")
                parts.append(f"{col}={total:.6g}")
            else:
                mode = Counter(vals).most_common(1)[0][0]
                parts.append(f"{col}={mode}")
        fold_strata[fold] = "||".join(parts)
    return fold_strata


def _assign_folds_to_train_test(
    fold_ids: list[str],
    *,
    seed: int,
    fold_strata: dict[str, str] | None,
    ratios: tuple[float, float, float] | None,
) -> dict[str, str]:
    if not fold_ids:
        return {}
    if len(fold_ids) < 3:
        labels = ["train", "val", "test"]
        out = {}
        for i, fid in enumerate(sorted(fold_ids)):
            out[fid] = labels[i % 3]
        return out
    rng = random.Random(seed)
    if fold_strata:
        strata = [fold_strata[f] for f in fold_ids]
        labels = assign_folds_stratified(fold_ids, strata, rng, ratios=ratios)
        return dict(zip(fold_ids, labels))
    order = list(fold_ids)
    rng.shuffle(order)
    labels = assign_folds_random(len(order), ratios=ratios)
    return {fid: lab for fid, lab in zip(order, labels)}


def assign_from_features(
    features: FeatureTable,
    *,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    seed: int = 42,
    n_clusters: int | Literal["auto"] = "auto",
    cluster_method: str | ClusterMethod = "dbscan",
    ratios: tuple[float, float, float] | None = None,
    precomputed_clusters: dict[str, int] | None = None,
    additional_by_region: dict[str, Any] | None = None,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int = 5,
    elbow_checkpoint_path: Path | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Contract C2: FeatureTable → assignment rows.

    Columns: region|cluster|train_test|fold|additional
    """
    cluster_method = normalize_cluster_method(cluster_method)
    fold_map = _load_id_fold_map(fold_csv)
    strat_map = _load_strat_map(stratification_csv)

    zsv_ids: list[str] = []
    assignable_ids: list[str] = []
    for rid in features.ids:
        raw = fold_map.get(rid, "0")
        if is_zsv_fold(raw):
            zsv_ids.append(rid)
        else:
            assignable_ids.append(rid)

    meta: dict[str, Any] = {
        "n_total": features.n,
        "n_zsv": len(zsv_ids),
        "n_assignable": len(assignable_ids),
        "seed": seed,
        "features": list(features.feature_names),
    }

    rows: list[dict[str, str]] = []
    for rid in zsv_ids:
        extra = ""
        if additional_by_region and rid in additional_by_region:
            extra = json.dumps(additional_by_region[rid], sort_keys=True)
        rows.append(
            {
                "region": rid,
                "cluster": "zsv",
                "train_test": "zsv",
                "fold": "zsv",
                "additional": extra,
            }
        )

    if not assignable_ids:
        meta["cluster"] = {"skipped": True}
        return rows, meta

    sub = features.subset(assignable_ids)
    # Drop parent matrix after subset copy so we do not hold two full n×d panels.
    large_panel = int(features.n) * int(features.n_features) >= 5_000_000
    if large_panel:
        object.__setattr__(
            features, "matrix", np.empty((0, 0), dtype=np.float32)
        )
    if precomputed_clusters is not None:
        cluster_map = {rid: int(precomputed_clusters[rid]) for rid in assignable_ids}
        cluster_meta = {"method_used": "precomputed"}
    else:
        cluster_map, cluster_meta = cluster_feature_table(
            sub,
            n_clusters=n_clusters,
            method=cluster_method,
            seed=seed,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            elbow_checkpoint_path=elbow_checkpoint_path,
        )
    meta["cluster"] = cluster_meta

    fold_members: dict[str, list[str]] = defaultdict(list)
    region_fold: dict[str, str] = {}
    for rid in assignable_ids:
        fold_label = str(cluster_map[rid])
        region_fold[rid] = fold_label
        fold_members[fold_label].append(rid)

    fold_strata = None
    if strat_map:
        missing = [rid for rid in assignable_ids if rid not in strat_map]
        if missing:
            raise ValueError(
                f"stratification.csv missing ID {missing[0]!r} "
                "(required when stratification is set)"
            )
        fold_strata = aggregate_stratification_per_fold(dict(fold_members), strat_map)
        meta["stratification"] = {
            "n_fold_strata": len(set(fold_strata.values())),
            "aggregated": True,
        }

    fold_to_tt = _assign_folds_to_train_test(
        sorted(fold_members),
        seed=seed,
        fold_strata=fold_strata,
        ratios=ratios,
    )
    meta["train_test_by_fold"] = fold_to_tt

    # Large panels: do NOT embed n×d floats into additional JSON (OOM / huge CSV).
    embed_features = not large_panel
    meta["additional_embeds_features"] = embed_features
    feat_index = {rid: i for i, rid in enumerate(sub.ids)}
    for rid in assignable_ids:
        fold_label = region_fold[rid]
        extra_obj: dict[str, Any] = {"cluster_method": cluster_meta.get("method_used")}
        if additional_by_region and rid in additional_by_region:
            extra_obj["user"] = additional_by_region[rid]
        if embed_features:
            ix = feat_index[rid]
            for j, name in enumerate(sub.feature_names):
                extra_obj[name] = float(sub.matrix[ix, j])
        rows.append(
            {
                "region": rid,
                "cluster": fold_label,
                "train_test": fold_to_tt[fold_label],
                "fold": fold_label,
                "additional": json.dumps(extra_obj, sort_keys=True),
            }
        )

    by_region = {r["region"]: r for r in rows}
    ordered = [by_region[rid] for rid in features.ids if rid in by_region]
    return ordered, meta


# Back-compat alias used by older call sites / docs
def assign_from_distance_matrix(*args: Any, **kwargs: Any) -> Any:
    raise TypeError(
        "assign_from_distance_matrix is retired for clustering; "
        "use assign_from_features(FeatureTable, ...) instead. "
        "Dense distances remain available only for small-n diagnostics."
    )


def assignment_rows_to_split_csv(
    rows: Sequence[dict[str, str]],
    outdir: Path,
    *,
    filename: str = "split.csv",
) -> Path:
    """Map SBS assignment → pipeline ``ID|train_test|fold`` split.csv."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    split_rows = [
        {"ID": r["region"], "train_test": r["train_test"], "fold": r["fold"]}
        for r in rows
    ]
    path = outdir / filename
    write_csv(path, split_rows, ["ID", "train_test", "fold"])
    assign_path = outdir / "sbs_assignment.csv"
    write_csv(assign_path, list(rows), ASSIGNMENT_COLUMNS)
    return path


def write_assignment_table(rows: Sequence[dict[str, str]], path: Path) -> Path:
    path = Path(path)
    write_csv(path, list(rows), ASSIGNMENT_COLUMNS)
    return path
