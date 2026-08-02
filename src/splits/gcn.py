"""GCN / VGAE labeling cascade → ``split.csv`` (see ``splits/GCN.md``).

Resolution order (LOCKED):
  1. Reuse existing ``split.csv`` / role scores for the named model
  2. Infer from checkpoint + pack (+ incidence for hash grain)
  3. Train VGCN/VGAE then label

Homology (OG/PG) never enters the encoder — reuse ``src.splits.vgae`` contracts.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.pipeline.common import read_csv, write_csv
from src.splits.sbs.assign import assignment_rows_to_split_csv
from src.splits.vgae.assign import assignment_rows, size_constrained_assign
from src.splits.vgae.graph_data import assert_no_homology_features, load_packed_graph
from src.splits.vgae.hash_export import pool_hash_scores_to_regions
from src.splits.vgae.model import ClassicVGAE, soft_role_probs

DEFAULT_VGAE_ROOT = Path("VGAE")
SPLIT_COLUMNS = ["ID", "train_test", "fold"]


class AmbiguousGcnModelError(ValueError):
    """Raised when a model description matches more than one VGAE run."""


class MissingGcnModelError(FileNotFoundError):
    """Raised when no VGAE/VGCN run matches the request and train inputs are absent."""


def _normalize_model_token(text: str) -> str:
    s = str(text).strip()
    s = s.replace("\\", "/")
    if s.startswith("VGAE/"):
        s = s[len("VGAE/") :]
    return s.strip("/").strip()


def _slug_from_description(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_")
    return slug or "gcn_model"


def list_vgae_model_dirs(vgae_root: Path | None = None) -> list[Path]:
    root = Path(vgae_root) if vgae_root is not None else DEFAULT_VGAE_ROOT
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "train_meta.json").is_file() or (p / "split.csv").is_file() or (
            p / "pack" / "feature_meta.json"
        ).is_file():
            out.append(p)
    return out


def resolve_gcn_model(
    model: str,
    *,
    vgae_root: Path | None = None,
    allow_create: bool = False,
) -> Path:
    """Resolve a model name or description to ``VGAE/<model>/``.

    Stops (raises) when zero matches and ``allow_create`` is False, or when
    more than one description match is found (missing-data-policy).
    """
    raw = str(model).strip()
    if not raw:
        raise ValueError("gcn model name/description must be non-empty")
    root = Path(vgae_root) if vgae_root is not None else DEFAULT_VGAE_ROOT
    token = _normalize_model_token(raw)

    # Exact path or child of VGAE root
    direct = Path(token)
    if direct.is_dir() and (
        (direct / "split.csv").is_file()
        or (direct / "train_meta.json").is_file()
        or (direct / "pack" / "feature_meta.json").is_file()
        or (direct / "checkpoints" / "best.pt").is_file()
    ):
        return direct.resolve()

    candidate = root / token
    if candidate.is_dir():
        return candidate.resolve()

    # Description / alias match against train_meta + directory names
    needle = raw.lower()
    hits: list[Path] = []
    for d in list_vgae_model_dirs(root):
        blob_parts = [d.name.lower()]
        meta_path = d / "train_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            for key in ("grain", "k", "stage", "device", "loss_mode"):
                if key in meta:
                    blob_parts.append(f"{key}={meta[key]}".lower())
            blob_parts.append(json.dumps(meta, sort_keys=True).lower())
        blob = " ".join(blob_parts)
        # Token overlap: every alphanumeric word in the query should appear
        words = [w for w in re.split(r"[^a-z0-9]+", needle) if len(w) >= 2]
        if not words:
            continue
        if all(w in blob or w in d.name.lower() for w in words):
            hits.append(d)

    # Prefer unique hit; if several share a common prefix name exact, keep unique names
    uniq = {h.resolve() for h in hits}
    if len(uniq) == 1:
        return next(iter(uniq))
    if len(uniq) > 1:
        names = ", ".join(sorted(p.name for p in uniq))
        raise AmbiguousGcnModelError(
            f"gcn model description {raw!r} matches multiple runs: {names}. "
            "Pass an exact VGAE/<name> path."
        )

    if allow_create:
        created = (root / _slug_from_description(token)).resolve()
        created.mkdir(parents=True, exist_ok=True)
        return created

    known = ", ".join(p.name for p in list_vgae_model_dirs(root)) or "(none)"
    raise MissingGcnModelError(
        f"no VGAE/VGCN model matched {raw!r} under {root}. "
        f"Known: {known}. Provide graph_dir+marked_dir to train a new run, "
        "or an exact model directory."
    )


def _has_labeling(model_dir: Path) -> bool:
    split_csv = model_dir / "split.csv"
    if not split_csv.is_file() or split_csv.stat().st_size == 0:
        return False
    rows = read_csv(split_csv)
    if not rows:
        return False
    required = {"ID", "train_test"}
    if not required.issubset(rows[0]):
        return False
    return True


def _has_infer_assets(model_dir: Path) -> bool:
    ckpt = model_dir / "checkpoints" / "best.pt"
    pack_meta = model_dir / "pack" / "feature_meta.json"
    return ckpt.is_file() and pack_meta.is_file()


def _copy_split_csv(src: Path, dest_dir: Path) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "split.csv"
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    # Ensure fold column exists
    rows = read_csv(dest)
    for row in rows:
        row.setdefault("fold", row.get("train_test", ""))
    write_csv(dest, rows, SPLIT_COLUMNS)
    return dest


def infer_gcn_labeling(
    model_dir: Path,
    *,
    outdir: Path | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    device: str | None = None,
    hidden_dim: int = 64,
    latent_dim: int = 32,
) -> Path:
    """Load checkpoint + pack, assign roles, write ``split.csv``."""
    model_dir = Path(model_dir)
    outdir = Path(outdir) if outdir is not None else model_dir
    outdir.mkdir(parents=True, exist_ok=True)
    pack = load_packed_graph(model_dir / "pack")
    assert_no_homology_features(pack.feature_names)

    ckpt_path = model_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    # Prefer dims encoded in the checkpoint over caller defaults
    if "conv1.lin.weight" in state:
        hidden_dim = int(state["conv1.lin.weight"].shape[0])
    if "conv_mu.lin.weight" in state:
        latent_dim = int(state["conv_mu.lin.weight"].shape[0])
    n_roles = 3
    if "role_head.weight" in state:
        n_roles = int(state["role_head.weight"].shape[0])

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    x = torch.as_tensor(pack.x, dtype=torch.float32, device=dev)
    edge_index = torch.stack(
        [
            torch.as_tensor(pack.edge_u, dtype=torch.long, device=dev),
            torch.as_tensor(pack.edge_v, dtype=torch.long, device=dev),
        ],
        dim=0,
    )
    edge_weight = torch.as_tensor(pack.edge_w, dtype=torch.float32, device=dev)
    model = ClassicVGAE(
        x.size(1),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        n_roles=n_roles,
    ).to(dev)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_weight)
        scores = soft_role_probs(out["role_logits"]).detach().cpu().numpy()

    grain = str(pack.meta.get("grain") or "region")
    if grain == "hash":
        inc_path = model_dir / "pack" / "incidence.npz"
        if not inc_path.is_file():
            raise FileNotFoundError(
                f"hash-grain model missing incidence.npz under {model_dir / 'pack'}"
            )
        with np.load(inc_path, allow_pickle=True) as data:
            incidence = {
                "indptr": data["indptr"],
                "indices": data["indices"],
                "region_ids": [str(x) for x in data["region_ids"].tolist()],
                "hash_values": data["hash_values"],
            }
        region_ids, region_scores = pool_hash_scores_to_regions(scores, incidence)
        labels = size_constrained_assign(region_scores, ratios=ratios, seed=seed)
        rows = assignment_rows(region_ids, labels, fold_prefix="gcn")
    else:
        labels = size_constrained_assign(scores, ratios=ratios, seed=seed)
        rows = assignment_rows(pack.ids, labels, fold_prefix="gcn")

    return assignment_rows_to_split_csv(rows, outdir)


def run_gcn_split_assign(
    *,
    outdir: Path,
    model: str,
    vgae_root: Path | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] | None = None,
    graph_dir: Path | None = None,
    marked_dir: Path | None = None,
    device: str | None = None,
    max_ids: int | None = None,
    k: int | None = None,
    force_retrain: bool = False,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """Cascade: reuse → infer → train; write ``outdir/split.csv``.

    Returns a summary dict with ``split_csv``, ``model_dir``, ``source`` ∈
    {``reuse``, ``infer``, ``train``}.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ratio_t = ratios if ratios is not None else (3.0, 1.0, 1.0)
    root = Path(vgae_root) if vgae_root is not None else DEFAULT_VGAE_ROOT

    can_train = graph_dir is not None and marked_dir is not None
    try:
        model_dir = resolve_gcn_model(
            model, vgae_root=root, allow_create=False
        )
    except MissingGcnModelError:
        if not can_train:
            raise
        model_dir = resolve_gcn_model(
            model, vgae_root=root, allow_create=True
        )

    # 1) Reuse labeling
    if not force_retrain and _has_labeling(model_dir):
        split_csv = _copy_split_csv(model_dir / "split.csv", outdir)
        return {
            "split_csv": str(split_csv),
            "model_dir": str(model_dir),
            "source": "reuse",
            "model": model,
        }

    # 2) Infer from checkpoint + pack
    if not force_retrain and _has_infer_assets(model_dir):
        split_csv = infer_gcn_labeling(
            model_dir,
            outdir=outdir,
            seed=seed,
            ratios=ratio_t,
            device=device,
        )
        # Also refresh model_dir labeling for later reuse
        if (outdir / "split.csv").resolve() != (model_dir / "split.csv").resolve():
            shutil.copy2(split_csv, model_dir / "split.csv")
        return {
            "split_csv": str(split_csv),
            "model_dir": str(model_dir),
            "source": "infer",
            "model": model,
        }

    # 3) Train then label
    if not can_train:
        raise MissingGcnModelError(
            f"model {model!r} at {model_dir} has no split.csv and no "
            f"checkpoint+pack; cannot train without graph_dir+marked_dir"
        )

    pack_meta = model_dir / "pack" / "feature_meta.json"
    grain = "region"
    if pack_meta.is_file():
        try:
            grain = str(
                json.loads(pack_meta.read_text(encoding="utf-8")).get("grain")
                or "region"
            )
        except json.JSONDecodeError:
            grain = "region"
    # Description heuristics for new runs
    desc = str(model).lower()
    if "hash" in desc and "region" not in desc:
        grain = "hash"

    if grain == "hash":
        from src.splits.vgae.stage2 import run_stage2_hash_vgae

        ids_file = Path(graph_dir) / "ids.txt"
        if not ids_file.is_file() and (Path(graph_dir) / "graph" / "ids.txt").is_file():
            ids_file = Path(graph_dir) / "graph" / "ids.txt"
        if not ids_file.is_file():
            raise FileNotFoundError(
                f"hash train requires graph ids.txt next to {graph_dir}"
            )
        region_ids = [
            ln.strip()
            for ln in ids_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        meta = run_stage2_hash_vgae(
            out_dir=model_dir,
            marked_dir=Path(marked_dir),
            region_ids=region_ids,
            k=int(k) if k is not None else 5,
            seed=seed,
            ratios=ratio_t,
            max_ids=max_ids,
            device=device,
            **train_kwargs,
        )
    else:
        from src.splits.vgae.split_assign import run_vgae_split_assign

        meta = run_vgae_split_assign(
            outdir=model_dir,
            graph_dir=Path(graph_dir),
            marked_dir=Path(marked_dir),
            seed=seed,
            k=k,
            max_ids=max_ids,
            ratios=ratio_t,
            device=device,
            **train_kwargs,
        )

    src_split = Path(meta["split_csv"])
    if not src_split.is_file():
        src_split = model_dir / "split.csv"
    if not src_split.is_file():
        raise FileNotFoundError(f"train did not produce split.csv under {model_dir}")
    split_csv = _copy_split_csv(src_split, outdir)
    return {
        "split_csv": str(split_csv),
        "model_dir": str(model_dir),
        "source": "train",
        "model": model,
        "train_meta": meta,
    }
