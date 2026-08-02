"""Homology ``sd_random`` objective — NEVER fed into the GCN encoder.

``L_hom = mean(sd_para) - mean(sd_ortho)`` (minimize → ortho together, para stratified).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


DEFAULT_HASH_TABLE = Path("mag/homology_graph/maps/gene_ortho_para_hash.tsv")


@dataclass(frozen=True)
class HomologyGroups:
    """Per-node group ids (-1 = unmapped) + inverted group → node index lists."""

    orthogroup: np.ndarray  # int64 (n,)
    paragroup: np.ndarray  # int64 (n,)
    ortho_groups: tuple[np.ndarray, ...]  # each array of node indices
    para_groups: tuple[np.ndarray, ...]


def _sd_random_np(
    counts: np.ndarray,
    fracs: np.ndarray,
) -> float:
    """counts shape (3,), fracs shape (3,)."""
    n = float(counts.sum())
    if n <= 0.0:
        return 0.0
    expected = n * fracs
    d = counts.astype(np.float64) - expected
    return float(np.sqrt(np.sum(d * d)))


def sd_random_from_labels(
    labels: Sequence[str],
    group_indices: Sequence[np.ndarray],
    *,
    role_order: tuple[str, str, str] = ("train", "test", "val"),
    max_groups: int | None = None,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Hard ``sd_random`` mean over groups; also return per-group values."""
    lab = np.asarray(list(labels), dtype=object)
    role_to_i = {r: i for i, r in enumerate(role_order)}
    role_codes = np.full(lab.shape[0], -1, dtype=np.int8)
    for r, i in role_to_i.items():
        role_codes[lab == r] = i
    global_counts = np.bincount(role_codes[role_codes >= 0], minlength=3).astype(
        np.float64
    )
    total = float(global_counts.sum())
    if total <= 0.0:
        return 0.0, []
    fracs = global_counts / total
    groups = [g for g in group_indices if g is not None and len(g) >= 2]
    if max_groups is not None and len(groups) > int(max_groups):
        rng = np.random.default_rng(int(seed))
        pick = rng.choice(len(groups), size=int(max_groups), replace=False)
        groups = [groups[int(i)] for i in pick.tolist()]
    vals: list[float] = []
    for idxs in groups:
        codes = role_codes[np.asarray(idxs, dtype=np.int64)]
        codes = codes[codes >= 0]
        if codes.size == 0:
            continue
        c = np.bincount(codes, minlength=3).astype(np.float64)
        vals.append(_sd_random_np(c, fracs))
    if not vals:
        return 0.0, []
    return float(np.mean(vals)), vals


def load_homology_groups(
    ids: Sequence[str],
    hash_table: Path | None = None,
) -> HomologyGroups:
    """Join panel IDs to orthogroup/paragroup via the MARKED hash table.

    Schema: ``id_MARKED|id_MARKED_hash|id|genome|orthogroup|orthogroup_hash|paragroup|paragroup_hash``
    """
    path = Path(hash_table) if hash_table is not None else DEFAULT_HASH_TABLE
    if not path.is_file():
        raise FileNotFoundError(f"homology hash table missing: {path}")

    id_to_idx = {str(rid): i for i, rid in enumerate(ids)}
    n = len(ids)
    ortho = np.full(n, -1, dtype=np.int64)
    para = np.full(n, -1, dtype=np.int64)
    ortho_key_to_gid: dict[str, int] = {}
    para_key_to_gid: dict[str, int] = {}

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        required = {"id_MARKED", "orthogroup", "paragroup"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            # allow positional fallback
            fh.seek(0)
            first = fh.readline()
            cols = first.strip().split("|")
            if "id_MARKED" not in cols:
                raise ValueError(
                    f"unexpected homology table header in {path}: {cols[:8]}"
                )
            fh.seek(0)
            reader = csv.DictReader(fh, delimiter="|")

        for row in reader:
            rid = (row.get("id_MARKED") or "").strip()
            if not rid or rid not in id_to_idx:
                continue
            i = id_to_idx[rid]
            og = (row.get("orthogroup") or "").strip()
            pg = (row.get("paragroup") or "").strip()
            if og:
                gid = ortho_key_to_gid.get(og)
                if gid is None:
                    gid = len(ortho_key_to_gid)
                    ortho_key_to_gid[og] = gid
                ortho[i] = gid
            if pg:
                gid = para_key_to_gid.get(pg)
                if gid is None:
                    gid = len(para_key_to_gid)
                    para_key_to_gid[pg] = gid
                para[i] = gid

    def _invert(labels: np.ndarray, n_groups: int) -> tuple[np.ndarray, ...]:
        buckets: list[list[int]] = [[] for _ in range(n_groups)]
        for i, g in enumerate(labels.tolist()):
            if g >= 0:
                buckets[int(g)].append(i)
        return tuple(np.asarray(b, dtype=np.int64) for b in buckets if len(b) >= 1)

    return HomologyGroups(
        orthogroup=ortho,
        paragroup=para,
        ortho_groups=_invert(ortho, len(ortho_key_to_gid)),
        para_groups=_invert(para, len(para_key_to_gid)),
    )


def write_homology_sidecar(
    path: Path,
    ids: Sequence[str],
    groups: HomologyGroups,
) -> Path:
    """Audit-only sidecar — must not be concatenated into encoder ``X``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ID|orthogroup_id|paragroup_id\n")
        for rid, og, pg in zip(ids, groups.orthogroup.tolist(), groups.paragroup.tolist()):
            fh.write(f"{rid}|{og}|{pg}\n")
    return path


def select_groups_epoch_stable(
    group_index_list: Sequence[np.ndarray],
    *,
    max_groups: int | None = 4096,
    seed: int = 0,
    min_size: int = 2,
) -> list[np.ndarray]:
    """Seeded epoch-stable group subset (additive; does not mutate callers).

    When ``len(groups) <= max_groups`` (or ``max_groups`` is None), returns all
    eligible groups. Otherwise samples a fixed subset for ``seed``.
    """
    groups = [
        g
        for g in group_index_list
        if g is not None and len(g) >= int(min_size)
    ]
    if not groups:
        return []
    if max_groups is None or len(groups) <= int(max_groups):
        return list(groups)
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(len(groups), size=int(max_groups), replace=False)
    return [groups[int(i)] for i in np.sort(pick).tolist()]


# Aggregation modes for per-group sd_random → scalar L_hom components.
# Legacy callers use mean / weighted; robust + log_balance are additive.
HOM_AGG_MEAN = "mean"
HOM_AGG_WEIGHTED = "weighted"
HOM_AGG_ROBUST = "robust"
HOM_AGG_LOG_BALANCE = "log_balance"
HOM_AGG_MODES = (
    HOM_AGG_MEAN,
    HOM_AGG_WEIGHTED,
    HOM_AGG_ROBUST,
    HOM_AGG_LOG_BALANCE,
)


def soft_sd_random_per_group(
    soft: torch.Tensor,
    group_index_list: Sequence[np.ndarray],
    *,
    max_groups: int | None = 4096,
    subset_seed: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``sd_random`` and sizes ``(G,)`` for soft role probs ``(n,3)``.

    Returns empty tensors when no eligible groups exist.
    """
    if soft.ndim != 2 or soft.size(1) != 3:
        raise ValueError(f"soft must be (n,3); got {tuple(soft.shape)}")
    if subset_seed is not None:
        groups = select_groups_epoch_stable(
            group_index_list, max_groups=max_groups, seed=int(subset_seed)
        )
    else:
        groups = [g for g in group_index_list if g is not None and len(g) >= 2]
        if max_groups is not None and len(groups) > int(max_groups):
            idx = torch.randperm(len(groups), generator=generator)[: int(max_groups)]
            groups = [groups[int(i)] for i in idx.tolist()]
    if not groups:
        empty = soft.new_zeros((0,))
        return empty, empty

    mass = soft.sum(dim=0)
    total = mass.sum().clamp_min(1e-12)
    fracs = mass / total

    pieces: list[torch.Tensor] = []
    seg_ids: list[torch.Tensor] = []
    for gi, idxs in enumerate(groups):
        idx_t = torch.as_tensor(idxs, device=soft.device, dtype=torch.long)
        pieces.append(soft.index_select(0, idx_t))
        seg_ids.append(
            torch.full((idx_t.numel(),), gi, device=soft.device, dtype=torch.long)
        )
    flat = torch.cat(pieces, dim=0)
    seg = torch.cat(seg_ids, dim=0)
    counts = torch.zeros((len(groups), 3), device=soft.device, dtype=soft.dtype)
    counts.index_add_(0, seg, flat)
    n = counts.sum(dim=1).clamp_min(1e-12)
    expected = n.unsqueeze(1) * fracs.unsqueeze(0)
    d = counts - expected
    sd = torch.sqrt(torch.sum(d * d, dim=1) + 1e-12)
    return sd, n


def aggregate_sd_values(
    sd: torch.Tensor,
    n: torch.Tensor,
    *,
    agg: str,
    weight_power: float = 0.5,
    winsor_lo: float = 0.10,
    winsor_hi: float = 0.90,
    log_eps: float = 1e-6,
) -> torch.Tensor:
    """Reduce per-group ``sd`` to a scalar under ``agg``.

    * ``mean`` — uniform mean of raw ``sd``
    * ``weighted`` — size-weighted mean ``w=n**weight_power``
    * ``robust`` — ``sd/√n`` then winsorized mean (quantile bounds detached)
    * ``log_balance`` — mean ``log10(sd + eps)`` (caller forms ortho−para)
    """
    mode = str(agg).lower().strip()
    if sd.numel() == 0:
        return sd.new_tensor(0.0)
    if mode == HOM_AGG_MEAN:
        return sd.mean()
    if mode == HOM_AGG_WEIGHTED:
        w = n.pow(float(weight_power)).clamp_min(1e-12)
        return (sd * w).sum() / w.sum()
    if mode == HOM_AGG_ROBUST:
        # Multinomial null: ||c−np|| scales ~√n — divide so large groups
        # cannot dominate the batch mean via absolute scale alone.
        z = sd / n.sqrt().clamp_min(1.0)
        if z.numel() >= 8:
            lo_q = float(max(0.0, min(0.49, float(winsor_lo))))
            hi_q = float(max(lo_q + 0.01, min(1.0, float(winsor_hi))))
            # Detach bounds: gradients flow for inliers; tails are clipped.
            lo = torch.quantile(z.detach(), lo_q)
            hi = torch.quantile(z.detach(), hi_q)
            z = torch.minimum(torch.maximum(z, lo), hi)
        return z.mean()
    if mode == HOM_AGG_LOG_BALANCE:
        return torch.log10(sd.clamp_min(float(log_eps))).mean()
    raise ValueError(
        f"unknown sd aggregation agg={agg!r}; expected one of {HOM_AGG_MODES}"
    )


def soft_sd_random(
    soft: torch.Tensor,
    group_index_list: Sequence[np.ndarray],
    *,
    max_groups: int | None = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable mean ``sd_random`` over groups given soft role probs ``(n,3)``.

    For large panels, samples up to ``max_groups`` groups per call (seeded via
    ``generator`` when provided) so the homology term stays tractable.

    Legacy uniform mean — prefer :func:`soft_sd_random_weighted` for homology-first.
    """
    sd, n = soft_sd_random_per_group(
        soft,
        group_index_list,
        max_groups=max_groups,
        subset_seed=None,
        generator=generator,
    )
    return aggregate_sd_values(sd, n, agg=HOM_AGG_MEAN)


def soft_sd_random_weighted(
    soft: torch.Tensor,
    group_index_list: Sequence[np.ndarray],
    *,
    max_groups: int | None = 4096,
    subset_seed: int | None = None,
    generator: torch.Generator | None = None,
    weight_power: float = 0.5,
) -> torch.Tensor:
    """Size-weighted mean ``sd_random`` with ``w_g = n_g ** weight_power`` (default √n).

    Additive companion to :func:`soft_sd_random`. When ``subset_seed`` is set,
    uses :func:`select_groups_epoch_stable` instead of per-call ``randperm``.
    """
    sd, n = soft_sd_random_per_group(
        soft,
        group_index_list,
        max_groups=max_groups,
        subset_seed=subset_seed,
        generator=generator,
    )
    return aggregate_sd_values(
        sd, n, agg=HOM_AGG_WEIGHTED, weight_power=float(weight_power)
    )


def soft_sd_random_agg(
    soft: torch.Tensor,
    group_index_list: Sequence[np.ndarray],
    *,
    agg: str,
    max_groups: int | None = 4096,
    subset_seed: int | None = None,
    generator: torch.Generator | None = None,
    weight_power: float = 0.5,
    winsor_lo: float = 0.10,
    winsor_hi: float = 0.90,
    log_eps: float = 1e-6,
) -> torch.Tensor:
    """Soft ``sd_random`` reduced by :func:`aggregate_sd_values`."""
    sd, n = soft_sd_random_per_group(
        soft,
        group_index_list,
        max_groups=max_groups,
        subset_seed=subset_seed,
        generator=generator,
    )
    return aggregate_sd_values(
        sd,
        n,
        agg=agg,
        weight_power=weight_power,
        winsor_lo=winsor_lo,
        winsor_hi=winsor_hi,
        log_eps=log_eps,
    )


def compute_l_hom(
    soft_or_labels,
    groups: HomologyGroups,
    *,
    soft: bool = True,
    weighted: bool = False,
    agg: str | None = None,
    max_groups: int | None = None,
    subset_seed: int | None = None,
    lambda_para: float = 1.0,
    lambda_ortho: float = 1.0,
    generator: torch.Generator | None = None,
    winsor_lo: float = 0.10,
    winsor_hi: float = 0.90,
    log_eps: float = 1e-6,
) -> dict[str, float | torch.Tensor]:
    """Return ``L_hom`` + components under the chosen SD aggregation.

    Default (``agg=None``, ``weighted=False``): legacy
    ``L_hom = λ_para·mean(sd_para) − λ_ortho·mean(sd_ortho)``.

    ``agg`` overrides ``weighted`` when set:

    * ``mean`` / ``weighted`` — classic / √n-weighted means of raw ``sd``
    * ``robust`` — winsorized mean of ``sd/√n`` (outlier-resistant)
    * ``log_balance`` — ``λ_o·E[log10(sd_ortho)] − λ_p·E[log10(sd_para)]``
      (≈ ``E[log10(sd_ortho/sd_para)]``; minimize → ortho tight, para spread)
    """
    lp = float(lambda_para)
    lo = float(lambda_ortho)
    if agg is None:
        mode = HOM_AGG_WEIGHTED if weighted else HOM_AGG_MEAN
    else:
        mode = str(agg).lower().strip()
        if mode not in HOM_AGG_MODES:
            raise ValueError(f"unknown agg={agg!r}; expected one of {HOM_AGG_MODES}")

    if soft:
        if not isinstance(soft_or_labels, torch.Tensor):
            raise TypeError("soft=True requires a torch.Tensor of role probs")
        mg = 4096 if max_groups is None else max_groups
        if mode == HOM_AGG_MEAN and agg is None and not weighted:
            # Exact legacy path (per-call randperm; ignore subset_seed)
            sd_ortho = soft_sd_random(
                soft_or_labels,
                groups.ortho_groups,
                max_groups=mg,
                generator=generator,
            )
            sd_para = soft_sd_random(
                soft_or_labels,
                groups.para_groups,
                max_groups=mg,
                generator=generator,
            )
        else:
            sd_ortho = soft_sd_random_agg(
                soft_or_labels,
                groups.ortho_groups,
                agg=mode,
                max_groups=mg,
                subset_seed=subset_seed,
                generator=generator,
                winsor_lo=winsor_lo,
                winsor_hi=winsor_hi,
                log_eps=log_eps,
            )
            sd_para = soft_sd_random_agg(
                soft_or_labels,
                groups.para_groups,
                agg=mode,
                max_groups=mg,
                subset_seed=(
                    None if subset_seed is None else int(subset_seed) + 1_000_003
                ),
                generator=generator,
                winsor_lo=winsor_lo,
                winsor_hi=winsor_hi,
                log_eps=log_eps,
            )
        if mode == HOM_AGG_LOG_BALANCE:
            # Minimize → small log sd_ortho, large log sd_para
            l_hom = lo * sd_ortho - lp * sd_para
        else:
            l_hom = lp * sd_para - lo * sd_ortho
        return {
            "l_hom": l_hom,
            "mean_sd_ortho": sd_ortho,
            "mean_sd_para": sd_para,
            "hom_agg": mode,
        }

    labels = list(soft_or_labels)
    max_g = 8192 if max_groups is None else int(max_groups)
    seed_o = 0 if subset_seed is None else int(subset_seed)
    seed_p = 1 if subset_seed is None else int(subset_seed) + 1_000_003
    mean_o, mean_p = _hard_sd_aggregate(
        labels,
        groups,
        agg=mode,
        max_groups=max_g,
        seed_o=seed_o,
        seed_p=seed_p,
        winsor_lo=winsor_lo,
        winsor_hi=winsor_hi,
        log_eps=log_eps,
    )
    if mode == HOM_AGG_LOG_BALANCE:
        l_hom_v = float(lo * mean_o - lp * mean_p)
    else:
        l_hom_v = float(lp * mean_p - lo * mean_o)
    return {
        "l_hom": l_hom_v,
        "mean_sd_ortho": float(mean_o),
        "mean_sd_para": float(mean_p),
        "hom_agg": mode,
    }


def _hard_sd_values(
    labels: Sequence[str],
    group_indices: Sequence[np.ndarray],
    *,
    role_order: tuple[str, str, str] = ("train", "test", "val"),
    max_groups: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard per-group ``sd_random`` values and sizes."""
    lab = np.asarray(list(labels), dtype=object)
    role_to_i = {r: i for i, r in enumerate(role_order)}
    role_codes = np.full(lab.shape[0], -1, dtype=np.int8)
    for r, i in role_to_i.items():
        role_codes[lab == r] = i
    global_counts = np.bincount(role_codes[role_codes >= 0], minlength=3).astype(
        np.float64
    )
    total = float(global_counts.sum())
    if total <= 0.0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    fracs = global_counts / total
    groups = select_groups_epoch_stable(
        group_indices, max_groups=max_groups, seed=int(seed)
    )
    vals: list[float] = []
    sizes: list[float] = []
    for idxs in groups:
        codes = role_codes[np.asarray(idxs, dtype=np.int64)]
        codes = codes[codes >= 0]
        if codes.size == 0:
            continue
        c = np.bincount(codes, minlength=3).astype(np.float64)
        n = float(c.sum())
        vals.append(_sd_random_np(c, fracs))
        sizes.append(n)
    if not vals:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.asarray(vals, dtype=np.float64), np.asarray(sizes, dtype=np.float64)


def _aggregate_sd_np(
    sd: np.ndarray,
    n: np.ndarray,
    *,
    agg: str,
    weight_power: float = 0.5,
    winsor_lo: float = 0.10,
    winsor_hi: float = 0.90,
    log_eps: float = 1e-6,
) -> float:
    """NumPy twin of :func:`aggregate_sd_values`."""
    mode = str(agg).lower().strip()
    if sd.size == 0:
        return 0.0
    if mode == HOM_AGG_MEAN:
        return float(sd.mean())
    if mode == HOM_AGG_WEIGHTED:
        w = np.power(n, float(weight_power))
        w = np.maximum(w, 1e-12)
        return float((sd * w).sum() / w.sum())
    if mode == HOM_AGG_ROBUST:
        z = sd / np.maximum(np.sqrt(n), 1.0)
        if z.size >= 8:
            lo_q = float(max(0.0, min(0.49, float(winsor_lo))))
            hi_q = float(max(lo_q + 0.01, min(1.0, float(winsor_hi))))
            lo = float(np.quantile(z, lo_q))
            hi = float(np.quantile(z, hi_q))
            z = np.clip(z, lo, hi)
        return float(z.mean())
    if mode == HOM_AGG_LOG_BALANCE:
        return float(np.log10(np.maximum(sd, float(log_eps))).mean())
    raise ValueError(f"unknown agg={agg!r}")


def _hard_sd_aggregate(
    labels: Sequence[str],
    groups: HomologyGroups,
    *,
    agg: str,
    max_groups: int | None,
    seed_o: int,
    seed_p: int,
    winsor_lo: float = 0.10,
    winsor_hi: float = 0.90,
    log_eps: float = 1e-6,
) -> tuple[float, float]:
    """Hard ortho/para aggregated SD under ``agg``."""
    if agg == HOM_AGG_MEAN:
        mean_o, _ = sd_random_from_labels(
            labels, groups.ortho_groups, max_groups=max_groups, seed=seed_o
        )
        mean_p, _ = sd_random_from_labels(
            labels, groups.para_groups, max_groups=max_groups, seed=seed_p
        )
        return float(mean_o), float(mean_p)
    sd_o, n_o = _hard_sd_values(
        labels, groups.ortho_groups, max_groups=max_groups, seed=seed_o
    )
    sd_p, n_p = _hard_sd_values(
        labels, groups.para_groups, max_groups=max_groups, seed=seed_p
    )
    return (
        _aggregate_sd_np(
            sd_o,
            n_o,
            agg=agg,
            winsor_lo=winsor_lo,
            winsor_hi=winsor_hi,
            log_eps=log_eps,
        ),
        _aggregate_sd_np(
            sd_p,
            n_p,
            agg=agg,
            winsor_lo=winsor_lo,
            winsor_hi=winsor_hi,
            log_eps=log_eps,
        ),
    )


def _hard_sd_random_weighted(
    labels: Sequence[str],
    group_indices: Sequence[np.ndarray],
    *,
    role_order: tuple[str, str, str] = ("train", "test", "val"),
    max_groups: int | None = None,
    seed: int = 42,
    weight_power: float = 0.5,
) -> float:
    """Hard size-weighted mean ``sd_random`` (audit / baselines)."""
    sd, n = _hard_sd_values(
        labels,
        group_indices,
        role_order=role_order,
        max_groups=max_groups,
        seed=seed,
    )
    return _aggregate_sd_np(sd, n, agg=HOM_AGG_WEIGHTED, weight_power=weight_power)


def _sd_side_balance(
    sd: np.ndarray,
    n: np.ndarray,
) -> dict[str, float]:
    """Skew / outlier-mass diagnostics for one homology side (ortho or para)."""
    if sd.size == 0:
        return {
            "n_groups": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "p90_over_median": 0.0,
            "mean_over_median": 0.0,
            "top5pct_mass_frac": 0.0,
            "top5pct_size_frac": 0.0,
            "mean_z": 0.0,
            "median_z": 0.0,
        }
    order = np.argsort(sd)
    sd_s = sd[order]
    n_s = n[order]
    med = float(np.median(sd_s))
    mean = float(sd_s.mean())
    p90 = float(np.quantile(sd_s, 0.90))
    p99 = float(np.quantile(sd_s, 0.99))
    top_k = max(1, int(np.ceil(0.05 * sd_s.size)))
    top = sd_s[-top_k:]
    top_n = n_s[-top_k:]
    z = sd / np.maximum(np.sqrt(n), 1.0)
    return {
        "n_groups": float(sd_s.size),
        "mean": mean,
        "median": med,
        "p90": p90,
        "p99": p99,
        "max": float(sd_s[-1]),
        "p90_over_median": float(p90 / med) if med > 1e-12 else float("inf"),
        "mean_over_median": float(mean / med) if med > 1e-12 else float("inf"),
        "top5pct_mass_frac": float(top.sum() / max(sd_s.sum(), 1e-12)),
        "top5pct_size_frac": float(top_n.sum() / max(n_s.sum(), 1e-12)),
        "mean_z": float(z.mean()),
        "median_z": float(np.median(z)),
    }


def sd_group_balance_report(
    labels: Sequence[str],
    groups: HomologyGroups,
    *,
    max_groups: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Dataset-wide SD balance: mean/median skew and top-5% outlier mass."""
    sd_o, n_o = _hard_sd_values(
        labels, groups.ortho_groups, max_groups=max_groups, seed=int(seed)
    )
    sd_p, n_p = _hard_sd_values(
        labels,
        groups.para_groups,
        max_groups=max_groups,
        seed=int(seed) + 1_000_003,
    )
    return {
        "ortho": _sd_side_balance(sd_o, n_o),
        "para": _sd_side_balance(sd_p, n_p),
        "legacy_l_hom": float(
            _aggregate_sd_np(sd_p, n_p, agg=HOM_AGG_MEAN)
            - _aggregate_sd_np(sd_o, n_o, agg=HOM_AGG_MEAN)
        ),
    }


def evaluate_split_all_aggs(
    labels: Sequence[str],
    groups: HomologyGroups,
    *,
    max_groups: int | None = None,
    seed: int = 42,
) -> dict[str, dict[str, float | str]]:
    """Hard ``L_hom`` under every aggregation (legacy compare on any split)."""
    out: dict[str, dict[str, float | str]] = {}
    for agg in HOM_AGG_MODES:
        h = compute_l_hom(
            labels,
            groups,
            soft=False,
            agg=agg,
            max_groups=max_groups,
            subset_seed=int(seed),
        )
        out[agg] = {
            "l_hom": float(h["l_hom"]),
            "mean_sd_ortho": float(h["mean_sd_ortho"]),
            "mean_sd_para": float(h["mean_sd_para"]),
            "hom_agg": str(h.get("hom_agg", agg)),
        }
    return out


class EmaTermNorm:
    """EMA of absolute term magnitudes for recon/KL rebalancing (additive)."""

    def __init__(self, *, decay: float = 0.9, eps: float = 1e-6) -> None:
        self.decay = float(decay)
        self.eps = float(eps)
        self.ema_recon: float | None = None
        self.ema_kl: float | None = None

    def update(self, recon: float, kl: float) -> tuple[float, float]:
        ar = abs(float(recon)) + self.eps
        ak = abs(float(kl)) + self.eps
        if self.ema_recon is None:
            self.ema_recon = ar
            self.ema_kl = ak
        else:
            d = self.decay
            self.ema_recon = d * self.ema_recon + (1.0 - d) * ar
            self.ema_kl = d * float(self.ema_kl) + (1.0 - d) * ak
        return float(self.ema_recon), float(self.ema_kl)

    def normalize(
        self, recon: torch.Tensor, kl: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Return ``(recon/ema, kl/ema, ema_recon, ema_kl)``; updates EMA from detach."""
        with torch.no_grad():
            er, ek = self.update(float(recon.detach().cpu()), float(kl.detach().cpu()))
        return recon / er, kl / ek, er, ek
