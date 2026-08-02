"""Homology ``sd_random`` objective — NEVER fed into the GCN encoder.

``L_hom = mean(sd_para) - mean(sd_ortho)`` (minimize → ortho together, para stratified).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    """
    if soft.ndim != 2 or soft.size(1) != 3:
        raise ValueError(f"soft must be (n,3); got {tuple(soft.shape)}")
    groups = [g for g in group_index_list if g is not None and len(g) >= 2]
    if not groups:
        return soft.new_tensor(0.0)
    if max_groups is not None and len(groups) > int(max_groups):
        # Deterministic subsample without inventing labels
        idx = torch.randperm(len(groups), generator=generator)[: int(max_groups)]
        groups = [groups[int(i)] for i in idx.tolist()]

    mass = soft.sum(dim=0)
    total = mass.sum().clamp_min(1e-12)
    fracs = mass / total

    # Concatenate selected groups with a segment id; reduce via index_add
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
    n = counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
    expected = n * fracs.unsqueeze(0)
    d = counts - expected
    return torch.sqrt(torch.sum(d * d, dim=1) + 1e-12).mean()


def compute_l_hom(
    soft_or_labels,
    groups: HomologyGroups,
    *,
    soft: bool = True,
) -> dict[str, float | torch.Tensor]:
    """Return ``L_hom = mean(sd_para) - mean(sd_ortho)`` plus components."""
    if soft:
        if not isinstance(soft_or_labels, torch.Tensor):
            raise TypeError("soft=True requires a torch.Tensor of role probs")
        sd_ortho = soft_sd_random(soft_or_labels, groups.ortho_groups)
        sd_para = soft_sd_random(soft_or_labels, groups.para_groups)
        l_hom = sd_para - sd_ortho
        return {
            "l_hom": l_hom,
            "mean_sd_ortho": sd_ortho,
            "mean_sd_para": sd_para,
        }

    labels = list(soft_or_labels)
    max_g = 8192
    mean_o, _ = sd_random_from_labels(
        labels, groups.ortho_groups, max_groups=max_g, seed=0
    )
    mean_p, _ = sd_random_from_labels(
        labels, groups.para_groups, max_groups=max_g, seed=1
    )
    return {
        "l_hom": float(mean_p - mean_o),
        "mean_sd_ortho": float(mean_o),
        "mean_sd_para": float(mean_p),
    }
