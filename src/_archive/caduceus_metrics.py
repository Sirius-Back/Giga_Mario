#!/usr/bin/env python3
"""Epoch metrics for Caduceus expression training — implements project `metrics.md`.

Use TorchMetrics for standard regression metrics; compute gene-/sample-wise
Pearson medians once per epoch over the full pred/target tensors.
Do NOT average batch-wise Pearson/Spearman/etc. manually.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    from torchmetrics import MetricCollection
    from torchmetrics.regression import (
        MeanAbsoluteError,
        MeanSquaredError,
        PearsonCorrCoef,
        R2Score,
        SpearmanCorrCoef,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchmetrics is required for Caduceus logging (see metrics.md). "
        "Install in caduceus_env: pip/conda install torchmetrics"
    ) from e

# Keys required every validation/train/test epoch (metrics.md)
EPOCH_METRIC_KEYS = (
    "loss",  # logged separately as val_loss / train_loss / test_loss by caller
    "pearson",
    "spearman",
    "mse",
    "rmse",
    "mae",
    "r2",
    "genewise_pearson_median",
    "samplewise_pearson_median",
)


def _disable_metric_dist_sync(metrics: MetricCollection) -> MetricCollection:
    """Disable TorchMetrics distributed sync (safe for rank-0 full-split CPU compute).

    Under NCCL, sync_on_compute=True + .to("cpu") triggers all_gather on CPU tensors
    → RuntimeError: No backend type associated with device type cpu.
    evaluate_split already gathers full tensors on rank 0; no metric dist sync needed.
    """
    for met in metrics.values():
        if hasattr(met, "sync_on_compute"):
            met.sync_on_compute = False
        if hasattr(met, "dist_sync_on_step"):
            met.dist_sync_on_step = False
    return metrics


def build_regression_metrics(device: torch.device | str | None = None) -> MetricCollection:
    """Recommended MetricCollection from metrics.md (+ r2).

    Per-metric sync_on_compute=False (torchmetrics 1.2.x: MetricCollection has no
    sync_on_compute kwarg). Callers compute once over full-split tensors on CPU /
    rank-0; avoids NCCL all_gather on CPU under torchrun.
    """
    metrics = MetricCollection(
        {
            "pearson": PearsonCorrCoef(sync_on_compute=False),
            "spearman": SpearmanCorrCoef(sync_on_compute=False),
            "mse": MeanSquaredError(sync_on_compute=False),
            "rmse": MeanSquaredError(squared=False, sync_on_compute=False),
            "mae": MeanAbsoluteError(sync_on_compute=False),
            "r2": R2Score(sync_on_compute=False),
        }
    )
    metrics = _disable_metric_dist_sync(metrics)
    if device is not None:
        metrics = metrics.to(device)
    return metrics


def genewise_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target : (N_samples, N_genes)
    returns per-gene Pearson (N_genes,)
    """
    pred = pred - pred.mean(dim=0)
    target = target - target.mean(dim=0)
    denom = torch.sqrt((pred**2).sum(dim=0)) * torch.sqrt((target**2).sum(dim=0)) + 1e-8
    return (pred * target).sum(dim=0) / denom


def samplewise_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target : (N_samples, N_genes)
    returns per-sample Pearson (N_samples,)
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    denom = torch.sqrt((pred**2).sum(dim=1)) * torch.sqrt((target**2).sum(dim=1)) + 1e-8
    return (pred * target).sum(dim=1) / denom


def _as_2d(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        return t.unsqueeze(1)
    if t.ndim == 2:
        return t
    raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(t.shape)}")


@torch.no_grad()
def compute_epoch_regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss: float | None = None,
    metrics: MetricCollection | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Compute full metrics.md suite once over an epoch's concatenated tensors.

    pred/target: (N,) scalar TPM or (N, G) multi-gene.
    """
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    flat_pred = pred.reshape(-1)
    flat_target = target.reshape(-1)
    # TorchMetrics Pearson/etc. expect matching 1D for global metrics
    coll = metrics or build_regression_metrics(device="cpu")
    coll = _disable_metric_dist_sync(coll)
    coll = coll.to("cpu")
    coll.reset()
    coll.update(flat_pred, flat_target)
    computed = coll.compute()
    out: dict[str, float] = {k: float(v.detach().cpu()) for k, v in computed.items()}
    coll.reset()

    pred2 = _as_2d(pred)
    target2 = _as_2d(target)
    gw = genewise_pearson(pred2, target2)
    out["genewise_pearson_median"] = float(gw.median().item()) if gw.numel() else float("nan")
    if pred2.shape[1] < 2:
        # sample-wise Pearson across <2 genes is undefined
        out["samplewise_pearson_median"] = float("nan")
    else:
        sw = samplewise_pearson(pred2, target2)
        out["samplewise_pearson_median"] = float(sw.median().item()) if sw.numel() else float("nan")

    if loss is not None:
        out["loss"] = float(loss)
    return out


def format_epoch_log(split: str, metrics: dict[str, Any], epoch: int | None = None) -> str:
    """One-line human log; prefix split (train/validation/test)."""
    prefix = f"epoch={epoch} " if epoch is not None else ""
    # Preferred order from metrics.md
    order = [
        "loss",
        "pearson",
        "spearman",
        "mse",
        "rmse",
        "mae",
        "r2",
        "genewise_pearson_median",
        "samplewise_pearson_median",
    ]
    parts = []
    for k in order:
        if k not in metrics:
            continue
        v = metrics[k]
        name = f"{split}_{k}" if k == "loss" else f"{split}_{k}"
        if k == "loss":
            name = f"{split}_loss" if not split.endswith("loss") else split
        try:
            parts.append(f"{name}={float(v):.6g}")
        except (TypeError, ValueError):
            parts.append(f"{name}={v}")
    for k, v in metrics.items():
        if k in order:
            continue
        parts.append(f"{split}_{k}={v}")
    return prefix + " ".join(parts)
