"""Size-constrained train/test/val assignment from VGAE role scores."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.splits.common import train_test_val_weights

ROLE_ORDER = ("train", "test", "val")  # matches logits dim 0,1,2


def role_target_fractions(
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
) -> np.ndarray:
    """Return fractions in ROLE_ORDER (train, test, val)."""
    w = train_test_val_weights(ratios)
    total = w["train"] + w["test"] + w["val"]
    return np.asarray(
        [w["train"] / total, w["test"] / total, w["val"] / total], dtype=np.float64
    )


def size_loss(soft: torch.Tensor, target_frac: torch.Tensor) -> torch.Tensor:
    """Penalize deviation of mean soft assignment from target fractions."""
    mean = soft.mean(dim=0)
    return torch.mean((mean - target_frac.to(dtype=soft.dtype, device=soft.device)) ** 2)


def size_constrained_assign(
    scores: np.ndarray | torch.Tensor,
    *,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    seed: int = 42,
) -> list[str]:
    """Hard assign nodes to train/test/val with exact size quotas (3:1:1 default).

    ``scores`` shape ``(n, 3)`` in ROLE_ORDER. Greedy: fill each role by
    descending score for that role, resolving conflicts by best margin, then
    repair leftovers into under-filled roles.
    """
    if isinstance(scores, torch.Tensor):
        s = scores.detach().cpu().numpy()
    else:
        s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 3:
        raise ValueError(f"scores must be (n, 3); got {s.shape}")
    n = int(s.shape[0])
    if n < 3:
        raise ValueError(f"need >=3 nodes; got {n}")

    fr = role_target_fractions(ratios)
    # Largest-remainder quotas
    raw = fr * n
    quotas = np.floor(raw).astype(int)
    rem = n - int(quotas.sum())
    order = np.argsort(-(raw - quotas))
    for i in range(rem):
        quotas[order[i % 3]] += 1
    for i in range(3):
        if quotas[i] < 1:
            donor = int(np.argmax(quotas))
            if quotas[donor] <= 1:
                raise ValueError(f"cannot allocate roles for n={n}")
            quotas[donor] -= 1
            quotas[i] = 1

    rng = np.random.default_rng(int(seed))
    # Tie-break noise
    s_work = s + rng.normal(0.0, 1e-8, size=s.shape)

    assigned = np.full(n, -1, dtype=np.int32)
    remaining = set(range(n))
    # Fill roles from largest quota to smallest for stability
    role_fill_order = list(np.argsort(-quotas))
    for role in role_fill_order:
        need = int(quotas[role])
        if need <= 0:
            continue
        cand = sorted(remaining, key=lambda i: (-s_work[i, role], i))
        take = cand[:need]
        for i in take:
            assigned[i] = role
            remaining.discard(i)

    # Any leftovers (shouldn't happen) → best remaining role with capacity
    if remaining:
        for i in sorted(remaining):
            for role in np.argsort(-s_work[i]):
                if int(np.sum(assigned == role)) < int(quotas[role]):
                    assigned[i] = int(role)
                    break
            else:
                assigned[i] = int(np.argmax(s_work[i]))

    # Final repair if counts mismatch (swap)
    for role in range(3):
        while int(np.sum(assigned == role)) > int(quotas[role]):
            members = np.where(assigned == role)[0]
            # Drop lowest-confidence member for this role
            drop = int(members[np.argmin(s_work[members, role])])
            # Move to most under-filled role
            counts = np.array([np.sum(assigned == r) for r in range(3)])
            deficits = quotas - counts
            deficits[role] = -10**9
            dest = int(np.argmax(deficits))
            assigned[drop] = dest

    labels = [ROLE_ORDER[int(a)] for a in assigned.tolist()]
    # Sanity
    for role, q in zip(ROLE_ORDER, quotas.tolist()):
        if labels.count(role) != q:
            # Soft fallback: re-slice by global ranking of train score
            order_idx = np.argsort(-s_work[:, 0])
            labels = [""] * n
            cursor = 0
            for role_name, qn in zip(ROLE_ORDER, quotas.tolist()):
                for j in range(qn):
                    labels[int(order_idx[cursor + j])] = role_name
                cursor += qn
            break
    return labels


def assignment_rows(
    ids: Sequence[str],
    labels: Sequence[str],
    *,
    fold_prefix: str = "vgae",
) -> list[dict[str, str]]:
    if len(ids) != len(labels):
        raise ValueError("ids/labels length mismatch")
    rows: list[dict[str, str]] = []
    for rid, lab in zip(ids, labels):
        rows.append(
            {
                "region": str(rid),
                "cluster": f"{fold_prefix}_{lab}",
                "train_test": str(lab),
                "fold": f"{fold_prefix}_{lab}",
                "additional": "",
            }
        )
    return rows
