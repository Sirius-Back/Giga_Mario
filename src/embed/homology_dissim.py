"""Within-group embedding dissimilarity for orthologs vs paralogs.

``D_hom_emb = mean(d_para) − mean(d_ortho)`` (↑ better: paralogs dissimilar,
orthologs cohesive). Cosine distance on train-fit centered-L2 LegNet layers.

Join: LegNet ``seq_id`` ``SAMPLE__N`` → MARKED id ``N`` → Compara OG/PG via
``load_homology_groups`` (same hash table as VGAE ``L_hom``).
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.embed import DEFAULT_LAYERS, ROLE_TRAIN
from src.embed.distances import fit_train_stats, transform_centered_l2
from src.embed.store import EmbedStore, load_store, mask_role
from src.pipeline.mem_guard import ensure_allocation_fits
from src.splits.vgae.homology_loss import (
    DEFAULT_HASH_TABLE,
    HomologyGroups,
    load_homology_groups,
    select_groups_epoch_stable,
)

DEFAULT_LAYERS_HOM = ("pooled", "stage0", "stage1_2", "head_h")
SEQ_ID_MARKED_RE = re.compile(r"^(.+)__(\d+)$")
RUN_NAME_RE = re.compile(r"^run\d+_legnet_(.+)$")


@dataclass(frozen=True)
class HomDissimScores:
    """Per-store homology embedding dissimilarity summary."""

    run: str
    split_method: str
    layer: str
    n_ids: int
    n_mapped: int
    coverage: float
    n_og: int
    n_pg: int
    mean_d_ortho: float
    mean_d_para: float
    sem_d_ortho: float
    sem_d_para: float
    D_hom_emb: float
    n_pairs_ortho: int
    n_pairs_para: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def marked_id_from_seq_id(seq_id: str) -> str | None:
    """Map LegNet ``SAMPLE__N`` (or bare numeric) → MARKED panel id string."""
    s = str(seq_id).strip()
    if not s:
        return None
    m = SEQ_ID_MARKED_RE.match(s)
    if m:
        return m.group(2)
    if s.isdigit():
        return s
    return None


def split_method_from_run(run_name: str) -> str:
    """``run2_legnet_random`` → ``random``; strip optional ``/foldK`` suffix."""
    s = str(run_name).strip()
    if "/fold" in s:
        s = s.split("/fold", 1)[0]
    m = RUN_NAME_RE.match(s)
    if m:
        return m.group(1)
    return s


def remap_groups_to_store_indices(
    ids: Sequence[str],
    groups: HomologyGroups,
) -> HomologyGroups:
    """``load_homology_groups`` was called on MARKED ids; rebuild store-index lists.

    ``groups.orthogroup[i]`` / ``paragroup[i]`` already align with store row ``i``
    when ``ids`` passed to ``load_homology_groups`` were MARKED strings in store
    order — so inverted lists are already store indices. This helper exists for
    the synthetic path where callers build ``HomologyGroups`` directly.
    """
    return groups


def _sem(vals: Sequence[float]) -> float:
    a = np.asarray(list(vals), dtype=np.float64)
    if a.size < 2:
        return float("nan") if a.size == 0 else 0.0
    return float(np.std(a, ddof=1) / np.sqrt(a.size))


def mean_pairwise_cosine_distance(x: np.ndarray) -> tuple[float, int]:
    """Mean upper-triangle cosine distance for L2-normalized rows.

    Returns ``(mean_d, n_pairs)``. Empty / singleton → ``(nan, 0)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        return float("nan"), 0
    n = int(x.shape[0])
    # Full Gram is fine for typical OG/PG sizes (≤ a few hundred).
    sims = x @ x.T
    np.clip(sims, -1.0, 1.0, out=sims)
    iu = np.triu_indices(n, k=1)
    d = 1.0 - sims[iu]
    return float(np.mean(d)), int(d.size)


def _subsample_members(
    idxs: np.ndarray,
    *,
    max_members: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    idxs = np.asarray(idxs, dtype=np.int64)
    if max_members is None or idxs.size <= int(max_members):
        return idxs
    pick = rng.choice(idxs.size, size=int(max_members), replace=False)
    return idxs[np.sort(pick)]


def group_mean_distances(
    x_unit: np.ndarray,
    group_index_list: Sequence[np.ndarray],
    *,
    max_groups: int | None = 8192,
    max_members: int | None = 256,
    seed: int = 42,
    min_size: int = 2,
) -> tuple[float, float, list[float], int]:
    """Mean / SEM of per-group mean pairwise cosine distances + total pairs."""
    groups = select_groups_epoch_stable(
        group_index_list,
        max_groups=max_groups,
        seed=seed,
        min_size=min_size,
    )
    rng = np.random.default_rng(int(seed) + 17)
    vals: list[float] = []
    n_pairs = 0
    for g in groups:
        idxs = _subsample_members(np.asarray(g, dtype=np.int64), max_members=max_members, rng=rng)
        if idxs.size < 2:
            continue
        d, npairs = mean_pairwise_cosine_distance(x_unit[idxs])
        if npairs <= 0 or not np.isfinite(d):
            continue
        vals.append(d)
        n_pairs += npairs
    if not vals:
        return float("nan"), float("nan"), [], 0
    return float(np.mean(vals)), _sem(vals), vals, n_pairs


@dataclass(frozen=True)
class MarkedHomologyMap:
    """Global MARKED id → orthogroup / paragroup string keys (once per hash table)."""

    ortho_key: dict[str, str]  # marked_id → orthogroup key
    para_key: dict[str, str]  # marked_id → paragroup key


def load_marked_homology_map(hash_table: Path | None = None) -> MarkedHomologyMap:
    """Read the Compara hash table once into MARKED→group-key dicts."""
    path = Path(hash_table) if hash_table is not None else DEFAULT_HASH_TABLE
    if not path.is_file():
        raise FileNotFoundError(f"homology hash table missing: {path}")
    ortho_key: dict[str, str] = {}
    para_key: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        if reader.fieldnames is None or "id_MARKED" not in set(reader.fieldnames):
            raise ValueError(f"unexpected homology table header in {path}")
        for row in reader:
            rid = (row.get("id_MARKED") or "").strip()
            if not rid:
                continue
            og = (row.get("orthogroup") or "").strip()
            pg = (row.get("paragroup") or "").strip()
            if og:
                ortho_key[rid] = og
            if pg:
                para_key[rid] = pg
    return MarkedHomologyMap(ortho_key=ortho_key, para_key=para_key)


def groups_from_marked_map(
    marked_ids: Sequence[str],
    hmap: MarkedHomologyMap,
) -> HomologyGroups:
    """Build store-index ``HomologyGroups`` from preloaded MARKED→key maps."""
    n = len(marked_ids)
    ortho = np.full(n, -1, dtype=np.int64)
    para = np.full(n, -1, dtype=np.int64)
    ortho_key_to_gid: dict[str, int] = {}
    para_key_to_gid: dict[str, int] = {}
    for i, mid in enumerate(marked_ids):
        if not mid:
            continue
        ok = hmap.ortho_key.get(str(mid))
        if ok:
            gid = ortho_key_to_gid.get(ok)
            if gid is None:
                gid = len(ortho_key_to_gid)
                ortho_key_to_gid[ok] = gid
            ortho[i] = gid
        pk = hmap.para_key.get(str(mid))
        if pk:
            gid = para_key_to_gid.get(pk)
            if gid is None:
                gid = len(para_key_to_gid)
                para_key_to_gid[pk] = gid
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


def load_groups_for_store_ids(
    ids: Sequence[str],
    hash_table: Path | None = None,
    *,
    hmap: MarkedHomologyMap | None = None,
) -> tuple[HomologyGroups, np.ndarray, int]:
    """Join store seq_ids → MARKED → OG/PG.

    Returns ``(groups, marked_ids_object_array, n_mapped)`` where ``groups``
    index lists are **store row indices**.
    """
    marked: list[str] = []
    for sid in ids:
        mid = marked_id_from_seq_id(str(sid))
        marked.append(mid if mid is not None else "")
    if hmap is None:
        # Legacy path: filter hash table to this id set (slower for many stores)
        groups = load_homology_groups(marked, hash_table=hash_table)
    else:
        groups = groups_from_marked_map(marked, hmap)
    n_mapped = int(((groups.orthogroup >= 0) | (groups.paragroup >= 0)).sum())
    return groups, np.asarray(marked, dtype=object), n_mapped


def score_store_layer(
    store: EmbedStore,
    layer: str,
    groups: HomologyGroups,
    *,
    max_groups: int | None = 8192,
    max_members: int | None = 256,
    seed: int = 42,
    run_name: str | None = None,
) -> HomDissimScores:
    """Compute ``D_hom_emb`` for one layer of an embed store."""
    if layer not in store.layers:
        raise KeyError(f"layer {layer!r} missing in {store.out_dir}")
    raw = np.asarray(store.layers[layer])
    n, d = int(raw.shape[0]), int(raw.shape[1])

    # Collect only indices that appear in sampled OG/PG groups (much smaller than N)
    og_sel = select_groups_epoch_stable(
        groups.ortho_groups, max_groups=max_groups, seed=seed, min_size=2
    )
    pg_sel = select_groups_epoch_stable(
        groups.para_groups, max_groups=max_groups, seed=seed + 1, min_size=2
    )
    need: set[int] = set()
    for g in og_sel:
        need.update(int(i) for i in np.asarray(g, dtype=np.int64).tolist())
    for g in pg_sel:
        need.update(int(i) for i in np.asarray(g, dtype=np.int64).tolist())
    # Always include train rows for centering stats (cap if huge)
    train_m = mask_role(store.roles, ROLE_TRAIN)
    train_idx = np.flatnonzero(train_m)
    if train_idx.size > 8192:
        rng_t = np.random.default_rng(int(seed) + 99)
        train_idx = rng_t.choice(train_idx, size=8192, replace=False)
    need.update(int(i) for i in train_idx.tolist())
    need_idx = np.asarray(sorted(need), dtype=np.int64)
    nbytes = int(need_idx.size) * d * 8 + 8 * 1024**2
    try:
        ensure_allocation_fits(
            nbytes,
            label=f"homology_dissim:{layer}",
            timeout_sec=300.0,
        )
    except MemoryError:
        # Fall back to a smaller member cap rather than dying mid-suite
        max_members = min(int(max_members or 256), 64)
        max_groups = min(int(max_groups or 8192), 2048)
        og_sel = select_groups_epoch_stable(
            groups.ortho_groups, max_groups=max_groups, seed=seed, min_size=2
        )
        pg_sel = select_groups_epoch_stable(
            groups.para_groups, max_groups=max_groups, seed=seed + 1, min_size=2
        )
        need = set()
        for g in og_sel:
            need.update(int(i) for i in np.asarray(g, dtype=np.int64).tolist())
        for g in pg_sel:
            need.update(int(i) for i in np.asarray(g, dtype=np.int64).tolist())
        need.update(int(i) for i in train_idx.tolist())
        need_idx = np.asarray(sorted(need), dtype=np.int64)
        ensure_allocation_fits(
            int(need_idx.size) * d * 8 + 8 * 1024**2,
            label=f"homology_dissim:{layer}:retry",
            timeout_sec=600.0,
        )

    raw_sub = np.asarray(raw[need_idx], dtype=np.float64)
    # Map store index → row in raw_sub
    pos = {int(i): j for j, i in enumerate(need_idx.tolist())}

    if train_idx.size >= 2:
        train_local = np.asarray([pos[int(i)] for i in train_idx.tolist()], dtype=np.int64)
        train_x = raw_sub[train_local]
    else:
        train_x = raw_sub
    stats = fit_train_stats(train_x)
    x_unit = transform_centered_l2(raw_sub, stats)

    def _remap_groups(glist: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        rng = np.random.default_rng(int(seed) + 17)
        for g in glist:
            idxs = np.asarray(g, dtype=np.int64)
            idxs = _subsample_members(idxs, max_members=max_members, rng=rng)
            local = np.asarray([pos[int(i)] for i in idxs.tolist() if int(i) in pos], dtype=np.int64)
            if local.size >= 2:
                out.append(local)
        return out

    mean_o, sem_o, _, np_o = group_mean_distances(
        x_unit,
        _remap_groups(og_sel),
        max_groups=None,  # already selected
        max_members=None,
        seed=seed,
    )
    mean_p, sem_p, _, np_p = group_mean_distances(
        x_unit,
        _remap_groups(pg_sel),
        max_groups=None,
        max_members=None,
        seed=seed + 1,
    )
    n_og = sum(1 for g in groups.ortho_groups if len(g) >= 2)
    n_pg = sum(1 for g in groups.para_groups if len(g) >= 2)
    n_mapped = int(((groups.orthogroup >= 0) | (groups.paragroup >= 0)).sum())
    coverage = float(n_mapped / n) if n else 0.0
    if np.isfinite(mean_o) and np.isfinite(mean_p):
        d_hom = float(mean_p - mean_o)
    else:
        d_hom = float("nan")

    name = run_name if run_name is not None else Path(store.out_dir).name
    return HomDissimScores(
        run=name,
        split_method=split_method_from_run(name),
        layer=layer,
        n_ids=n,
        n_mapped=n_mapped,
        coverage=coverage,
        n_og=int(n_og),
        n_pg=int(n_pg),
        mean_d_ortho=mean_o,
        mean_d_para=mean_p,
        sem_d_ortho=sem_o,
        sem_d_para=sem_p,
        D_hom_emb=d_hom,
        n_pairs_ortho=int(np_o),
        n_pairs_para=int(np_p),
    )


def discover_embed_stores(base: Path) -> list[Path]:
    """Return store dirs under ``base`` that have ``manifest.json`` + ``ids.npy``."""
    base = Path(base)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if (child / "manifest.json").is_file() and (child / "ids.npy").is_file():
            out.append(child)
            continue
        # fold subdirs
        for sub in sorted(child.iterdir()):
            if (
                sub.is_dir()
                and (sub / "manifest.json").is_file()
                and (sub / "ids.npy").is_file()
            ):
                out.append(sub)
    return out


def score_store(
    store_dir: Path,
    *,
    layers: Iterable[str] = DEFAULT_LAYERS_HOM,
    hash_table: Path | None = None,
    max_groups: int | None = 8192,
    max_members: int | None = 256,
    seed: int = 42,
) -> list[HomDissimScores]:
    store_dir = Path(store_dir)
    layer_list = tuple(layers)
    store = load_store(store_dir, layers=layer_list)
    groups, _, _ = load_groups_for_store_ids(store.ids.tolist(), hash_table=hash_table)
    run_name = store_dir.name
    # Prefer manifest run_name when present
    man = store_dir / "manifest.json"
    if man.is_file():
        try:
            payload = json.loads(man.read_text(encoding="utf-8"))
            run_name = str(payload.get("run_name") or payload.get("run_key") or run_name)
        except (OSError, json.JSONDecodeError):
            pass
    return [
        score_store_layer(
            store,
            layer,
            groups,
            max_groups=max_groups,
            max_members=max_members,
            seed=seed,
            run_name=run_name,
        )
        for layer in layer_list
        if layer in store.layers
    ]


PER_STORE_COLUMNS = (
    "run",
    "split_method",
    "layer",
    "n_ids",
    "n_mapped",
    "coverage",
    "n_og",
    "n_pg",
    "mean_d_ortho",
    "mean_d_para",
    "sem_d_ortho",
    "sem_d_para",
    "D_hom_emb",
    "n_pairs_ortho",
    "n_pairs_para",
)


def write_per_store_tsv(rows: Sequence[HomDissimScores | dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [r.to_row() if isinstance(r, HomDissimScores) else dict(r) for r in rows]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(PER_STORE_COLUMNS), delimiter="\t")
        w.writeheader()
        for d in dicts:
            w.writerow({k: d.get(k, "") for k in PER_STORE_COLUMNS})
    return path


def write_ranking_tsv(
    rows: Sequence[HomDissimScores | dict[str, Any]],
    path: Path,
    *,
    layer: str = "pooled",
) -> Path:
    """Sort by ``D_hom_emb`` descending for one primary layer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [r.to_row() if isinstance(r, HomDissimScores) else dict(r) for r in rows]
    subset = [d for d in dicts if str(d.get("layer")) == layer]
    subset.sort(
        key=lambda d: (
            -float(d["D_hom_emb"]) if np.isfinite(float(d.get("D_hom_emb", "nan"))) else 1e9,
            str(d.get("run", "")),
        )
    )
    cols = list(PER_STORE_COLUMNS) + ["rank"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for i, d in enumerate(subset, start=1):
            row = {k: d.get(k, "") for k in PER_STORE_COLUMNS}
            row["rank"] = i
            w.writerow(row)
    return path


def run_all_stores(
    embed_root: Path,
    out_dir: Path,
    *,
    layers: Iterable[str] = DEFAULT_LAYERS_HOM,
    hash_table: Path | None = None,
    max_groups: int | None = 8192,
    max_members: int | None = 256,
    seed: int = 42,
    primary_layer: str = "pooled",
) -> dict[str, Path]:
    """Score every store; write per-layer TSVs + ranking."""
    embed_root = Path(embed_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hash_path = Path(hash_table) if hash_table is not None else DEFAULT_HASH_TABLE
    if not hash_path.is_file():
        raise FileNotFoundError(f"homology hash table missing: {hash_path}")

    stores = discover_embed_stores(embed_root)
    if not stores:
        raise FileNotFoundError(f"no embed stores under {embed_root}")

    layer_list = tuple(layers)
    all_rows: list[HomDissimScores] = []
    print(f"[homology_dissim] loading MARKED homology map from {hash_path}", flush=True)
    hmap = load_marked_homology_map(hash_path)
    print(
        f"[homology_dissim] map size ortho={len(hmap.ortho_key)} para={len(hmap.para_key)}",
        flush=True,
    )

    for store_dir in stores:
        print(f"[homology_dissim] scoring {store_dir.name}", flush=True)
        # ids/roles only first (no layers) — load layers one-by-one for RAM
        ids = np.load(store_dir / "ids.npy", allow_pickle=True)
        roles = np.load(store_dir / "roles.npy")
        groups, _, _ = load_groups_for_store_ids(
            ids.tolist(), hash_table=hash_path, hmap=hmap
        )

        run_name = store_dir.name
        man = store_dir / "manifest.json"
        if man.is_file():
            try:
                payload = json.loads(man.read_text(encoding="utf-8"))
                run_name = str(
                    payload.get("run_name") or payload.get("run_key") or run_name
                )
            except (OSError, json.JSONDecodeError):
                pass
        # LOO fold stores: keep fold suffix in run label for disambiguation
        if store_dir.name.startswith("fold") and store_dir.parent.name not in (
            "",
            ".",
        ):
            run_name = f"{store_dir.parent.name}/{store_dir.name}"

        for layer in layer_list:
            layer_path = store_dir / f"layer_{layer}.npy"
            if not layer_path.is_file():
                continue
            mat = np.load(layer_path, mmap_mode="r")
            store = EmbedStore(
                out_dir=store_dir,
                ids=ids,
                roles=roles,
                layers={layer: mat},
            )
            all_rows.append(
                score_store_layer(
                    store,
                    layer,
                    groups,
                    max_groups=max_groups,
                    max_members=max_members,
                    seed=seed,
                    run_name=run_name,
                )
            )
            del store, mat

    written: dict[str, Path] = {}
    by_layer: dict[str, list[HomDissimScores]] = {}
    for r in all_rows:
        by_layer.setdefault(r.layer, []).append(r)
    for layer, rows in by_layer.items():
        p = write_per_store_tsv(rows, out_dir / f"per_store_{layer}.tsv")
        written[f"per_store_{layer}"] = p
    ranking = write_ranking_tsv(all_rows, out_dir / "ranking.tsv", layer=primary_layer)
    written["ranking"] = ranking

    meta = {
        "embed_root": str(embed_root),
        "hash_table": str(hash_path),
        "n_stores": len(stores),
        "layers": list(layers),
        "primary_layer": primary_layer,
        "max_groups": max_groups,
        "max_members": max_members,
        "seed": seed,
        "metric": "centered_cosine_distance",
        "formula": "D_hom_emb = mean_d_para - mean_d_ortho",
        "stores": [str(s) for s in stores],
    }
    meta_path = out_dir / "manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    written["manifest"] = meta_path
    return written


__all__ = [
    "DEFAULT_LAYERS_HOM",
    "HomDissimScores",
    "MarkedHomologyMap",
    "discover_embed_stores",
    "group_mean_distances",
    "groups_from_marked_map",
    "load_groups_for_store_ids",
    "load_marked_homology_map",
    "marked_id_from_seq_id",
    "mean_pairwise_cosine_distance",
    "remap_groups_to_store_indices",
    "run_all_stores",
    "score_store",
    "score_store_layer",
    "split_method_from_run",
    "write_per_store_tsv",
    "write_ranking_tsv",
]
