"""Train-fit transforms and pairwise similarities for leakage analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-8
DEFAULT_RIDGE = 1e-3


@dataclass
class TrainStats:
    """Statistics fit on train embeddings only."""

    mean: np.ndarray  # [D]
    cov: np.ndarray | None = None  # [D, D]
    whiten: np.ndarray | None = None  # [D, D]  W such that x @ W
    cond: float | None = None
    ridge: float = DEFAULT_RIDGE

    def to_npz(self, path: Path) -> None:
        path = Path(path)
        payload: dict[str, Any] = {
            "mean": self.mean.astype(np.float64),
            "ridge": np.asarray(self.ridge),
        }
        if self.cov is not None:
            payload["cov"] = self.cov.astype(np.float64)
        if self.whiten is not None:
            payload["whiten"] = self.whiten.astype(np.float64)
        if self.cond is not None:
            payload["cond"] = np.asarray(self.cond)
        np.savez_compressed(path, **payload)

    @classmethod
    def from_npz(cls, path: Path) -> TrainStats:
        z = np.load(Path(path))
        return cls(
            mean=np.asarray(z["mean"], dtype=np.float64),
            cov=np.asarray(z["cov"], dtype=np.float64) if "cov" in z else None,
            whiten=np.asarray(z["whiten"], dtype=np.float64) if "whiten" in z else None,
            cond=float(z["cond"]) if "cond" in z else None,
            ridge=float(z["ridge"]) if "ridge" in z else DEFAULT_RIDGE,
        )


def fit_train_stats(
    train_x: np.ndarray, *, ridge: float = DEFAULT_RIDGE
) -> TrainStats:
    """Fit mean + ridge whitening on train ``[N, D]``."""
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [N,D], got {x.shape}")
    n, d = x.shape
    if n < 2:
        mean = x.mean(axis=0) if n else np.zeros(d, dtype=np.float64)
        return TrainStats(mean=mean, ridge=ridge)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / max(n - 1, 1)
    # Symmetrize for numerical stability
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(d, dtype=np.float64)
    try:
        evals = np.linalg.eigvalsh(cov)
        cond = float(evals.max() / max(evals.min(), EPS)) if evals.size else None
    except np.linalg.LinAlgError:
        cond = None
    # ZCA whitening: (Σ + εI)^{-1/2}
    try:
        evals_r, evecs = np.linalg.eigh(cov + ridge * eye)
        evals_r = np.clip(evals_r, EPS, None)
        whiten = evecs @ np.diag(1.0 / np.sqrt(evals_r)) @ evecs.T
    except np.linalg.LinAlgError:
        whiten = eye
        cond = float("inf")
    return TrainStats(mean=mean, cov=cov, whiten=whiten, cond=cond, ridge=ridge)


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def transform_centered_l2(x: np.ndarray, stats: TrainStats) -> np.ndarray:
    return _l2_normalize(np.asarray(x, dtype=np.float64) - stats.mean)


def transform_whitened_l2(x: np.ndarray, stats: TrainStats) -> np.ndarray:
    if stats.whiten is None:
        raise ValueError("TrainStats.whiten is missing")
    y = (np.asarray(x, dtype=np.float64) - stats.mean) @ stats.whiten
    return _l2_normalize(y)


def transform_row_centered_l2(x: np.ndarray) -> np.ndarray:
    """Per-vector centering then L2 → correlation / Pearson geometry."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    return _l2_normalize(x)


METRICS = (
    "centered_cosine",
    "l2_euclidean",
    "corr_distance",
    "whitened_cosine",
)


def prepare_metric_matrix(
    x: np.ndarray, stats: TrainStats, metric: str
) -> np.ndarray:
    """Map raw embeddings to unit vectors (or rows) for the chosen metric."""
    if metric == "centered_cosine":
        return transform_centered_l2(x, stats)
    if metric == "l2_euclidean":
        return transform_centered_l2(x, stats)
    if metric == "corr_distance":
        return transform_row_centered_l2(x)
    if metric == "whitened_cosine":
        return transform_whitened_l2(x, stats)
    raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")


def pairwise_max_similarity(
    query: np.ndarray,
    gallery: np.ndarray,
    *,
    metric: str,
    chunk: int = 4096,
    device: str | None = None,
) -> np.ndarray:
    """For each query row, max cosine similarity vs gallery (float32).

    Prepared matrices are L2-normalized → cosine = q @ g.T.
    For ``l2_euclidean`` on unit vectors this ranking is monotone-equivalent
    to Euclidean; we still return cosine values for a shared [−1,1] scale.
    Uses CUDA via torch when available (or ``device`` is set).
    """
    q = np.asarray(query, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError("query/gallery must be 2D")
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"dim mismatch {q.shape} vs {g.shape}")
    n_q = q.shape[0]
    out = np.empty(n_q, dtype=np.float32)
    if n_q == 0:
        return out.astype(np.float64)
    if g.shape[0] == 0:
        out[:] = -np.inf
        return out.astype(np.float64)

    use_torch = False
    torch_dev = "cpu"
    try:
        import torch

        if device is not None:
            torch_dev = device
            use_torch = True
        elif torch.cuda.is_available():
            torch_dev = "cuda"
            use_torch = True
    except ImportError:
        use_torch = False

    if use_torch:
        import torch

        g_t = torch.from_numpy(g).to(torch_dev).T.contiguous()
        for start in range(0, n_q, chunk):
            sl = torch.from_numpy(q[start : start + chunk]).to(torch_dev)
            sims = sl @ g_t
            out[start : start + len(sl)] = sims.max(dim=1).values.detach().cpu().numpy()
        del g_t
        if torch_dev.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        g_t = g.T
        for start in range(0, n_q, chunk):
            sl = q[start : start + chunk]
            sims = sl @ g_t
            out[start : start + chunk] = sims.max(axis=1)
    _ = metric  # ranking-equivalent families share cosine values
    return out.astype(np.float64)


def cosine_self_similarity(x: np.ndarray, stats: TrainStats, metric: str) -> float:
    """Sanity: first row vs itself after transform should be ~1."""
    y = prepare_metric_matrix(x[:1], stats, metric)
    return float((y @ y.T)[0, 0])
