"""Stage-2 hash-graph VGAE train → region-level split.csv."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.splits.sbs.assign import assignment_rows_to_split_csv
from src.splits.vgae.assign import assignment_rows, size_constrained_assign
from src.splits.vgae.graph_data import assert_no_homology_features
from src.splits.vgae.hash_export import pack_hash_graph, pool_hash_scores_to_regions
from src.splits.vgae.homology_loss import compute_l_hom, load_homology_groups, write_homology_sidecar
from src.splits.vgae.model import ClassicVGAE, soft_role_probs
from src.splits.vgae.train import (
    _append_status,
    _pick_free_gpu,
    _write_epoch_logs,
    run_vgae_train,
)
from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry, wait_until_launchable
from src.splits.vgae.assign import role_target_fractions, size_loss
from src.tb_logging import close_dual, log_scalar_pair, open_summary_writer, open_tensorboard_logger


def run_stage2_hash_vgae(
    *,
    out_dir: Path,
    marked_dir: Path,
    region_ids: list[str] | Path,
    k: int = 5,
    seed: int = 42,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    max_ids: int | None = None,
    max_edges: int = 500_000,
    device: str | None = None,
    homology_table: Path | None = None,
    min_epochs: int = 25,
    patience: int = 10,
    max_epochs: int = 200,
    peak_ram_gib: float = 16.0,
    wait_poll_sec: float = 600.0,
    **train_kw: Any,
) -> dict[str, Any]:
    """Export hash graph, train VGAE on hash nodes, pool → region split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = out_dir / "pack"

    if isinstance(region_ids, Path):
        region_ids = [
            ln.strip()
            for ln in Path(region_ids).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

    from src.splits.vgae.graph_data import load_packed_graph
    import numpy as np

    if (pack_dir / "feature_meta.json").is_file() and (pack_dir / "incidence.npz").is_file():
        pack = load_packed_graph(pack_dir)
        with np.load(pack_dir / "incidence.npz", allow_pickle=True) as data:
            incidence = {
                "indptr": data["indptr"],
                "indices": data["indices"],
                "region_ids": [str(x) for x in data["region_ids"].tolist()],
                "hash_values": data["hash_values"],
            }
        print(f"[vgae-s2] reusing existing pack under {pack_dir}", flush=True)
    else:
        pack, incidence = pack_hash_graph(
            Path(marked_dir),
            region_ids,
            pack_dir,
            k=k,
            max_edges=max_edges,
            max_ids=max_ids,
        )
    assert_no_homology_features(pack.feature_names)

    # Train on hash nodes but evaluate L_hom on pooled region assignments
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    if device is None:
        gpu_idx = _pick_free_gpu()
        wait_until_launchable(
            peak_ram_gib=peak_ram_gib,
            gpus=(gpu_idx,),
            job_class=CLASS_GPU_TRAIN,
            timeout_sec=6 * 3600,
            poll_sec=wait_poll_sec,
            label=f"vgae_stage2:{out_dir.name}",
        )
        gpu_idx = _pick_free_gpu()
        device = f"cuda:{gpu_idx}"
    else:
        gpu_idx = int(device.split(":")[1]) if ":" in device else 0
        wait_until_launchable(
            peak_ram_gib=peak_ram_gib,
            gpus=(gpu_idx,),
            job_class=CLASS_GPU_TRAIN,
            timeout_sec=6 * 3600,
            poll_sec=wait_poll_sec,
            label=f"vgae_stage2:{out_dir.name}",
        )

    queue_name = f"vgae_{out_dir.name}"
    append_queue_entry(
        queue_name,
        job=f"stage2_hash_vgae out={out_dir}",
        pid=os.getpid(),
        estimated_time="2-6h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=peak_ram_gib,
        gpus=(gpu_idx,),
        log=str(out_dir / "logs" / "metrics.log"),
        resources=f"device={device} n_hash={pack.n_nodes}",
    )

    try:
        region_id_list = [str(x) for x in incidence["region_ids"]]
        groups = load_homology_groups(region_id_list, homology_table)
        write_homology_sidecar(
            out_dir / "pack" / "node_homology_regions.tsv", region_id_list, groups
        )

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
            hidden_dim=int(train_kw.get("hidden_dim", 64)),
            latent_dim=int(train_kw.get("latent_dim", 32)),
        ).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=float(train_kw.get("lr", 1e-3)))
        beta_kl = float(train_kw.get("beta_kl", 1.0))
        lambda_hom = float(train_kw.get("lambda_hom", 1.0))
        lambda_size = float(train_kw.get("lambda_size", 1.0))
        target_frac = torch.as_tensor(
            role_target_fractions(ratios), dtype=torch.float32, device=dev
        )

        writer = open_summary_writer(out_dir)
        tb_logger = open_tensorboard_logger(out_dir)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        n_r = len(region_id_list)
        indptr_np = np.asarray(incidence["indptr"], dtype=np.int32)
        indices_np = np.asarray(incidence["indices"], dtype=np.int32)
        # Expand CSR → (region_idx, hash_idx) for vectorized mean-pool
        region_of_inc = np.repeat(
            np.arange(n_r, dtype=np.int64), np.diff(indptr_np).astype(np.int64)
        )
        region_idx_t = torch.as_tensor(region_of_inc, device=dev, dtype=torch.long)
        hash_idx_t = torch.as_tensor(indices_np, device=dev, dtype=torch.long)
        ones_inc = torch.ones((hash_idx_t.numel(), 1), device=dev, dtype=torch.float32)

        best_l_hom = float("inf")
        best_state = None
        best_epoch = -1
        stale = 0
        epoch_rows: list[dict[str, Any]] = []

        for epoch in range(1, int(max_epochs) + 1):
            model.train()
            out = model(x, edge_index, edge_weight)
            soft_h = soft_role_probs(out["role_logits"])
            # Differentiable mean-pool hash → region (homology stays out of encoder)
            pooled = torch.zeros((n_r, 3), device=dev, dtype=soft_h.dtype)
            counts = torch.zeros((n_r, 1), device=dev, dtype=soft_h.dtype)
            if hash_idx_t.numel() > 0:
                pooled.index_add_(0, region_idx_t, soft_h.index_select(0, hash_idx_t))
                counts.index_add_(0, region_idx_t, ones_inc.to(dtype=soft_h.dtype))
            pooled = pooled / counts.clamp_min(1.0)
            empty = (counts.squeeze(-1) <= 0)
            if empty.any():
                pooled = pooled.clone()
                pooled[empty] = soft_h.new_tensor([1.0, 0.0, 0.0])
            pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            recon = ClassicVGAE.recon_loss_neg_sample(out["z"], edge_index, edge_weight)
            kl = ClassicVGAE.kl_loss(out["mu"], out["logstd"])
            hom = compute_l_hom(pooled, groups, soft=True)
            sz = size_loss(pooled, target_frac)
            loss = recon + beta_kl * kl + lambda_hom * hom["l_hom"] + lambda_size * sz
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_e = model(x, edge_index, edge_weight)
                hash_scores = soft_role_probs(out_e["role_logits"]).cpu().numpy()
            rids, region_scores = pool_hash_scores_to_regions(hash_scores, incidence)
            labels = size_constrained_assign(region_scores, ratios=ratios, seed=seed + epoch)
            hard = compute_l_hom(labels, groups, soft=False)
            l_hom_hard = float(hard["l_hom"])
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "recon": float(recon.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "l_hom_soft": float(hom["l_hom"].detach().cpu()),
                "size": float(sz.detach().cpu()),
                "l_hom_hard": l_hom_hard,
                "mean_sd_ortho": float(hard["mean_sd_ortho"]),
                "mean_sd_para": float(hard["mean_sd_para"]),
            }
            epoch_rows.append(row)
            _write_epoch_logs(out_dir, epoch_rows)
            log_scalar_pair(writer, tb_logger, "train/loss", row["loss"], epoch)
            log_scalar_pair(writer, tb_logger, "validation/l_hom", l_hom_hard, epoch)
            log_scalar_pair(
                writer, tb_logger, "validation/mean_sd_ortho", row["mean_sd_ortho"], epoch
            )
            log_scalar_pair(
                writer, tb_logger, "validation/mean_sd_para", row["mean_sd_para"], epoch
            )

            if l_hom_hard < best_l_hom - 1e-6:
                best_l_hom = l_hom_hard
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {"epoch": epoch, "model": best_state, "l_hom": best_l_hom},
                    ckpt_dir / "best.pt",
                )
                stale = 0
            elif epoch >= int(min_epochs):
                stale += 1

            print(
                f"[vgae-s2] epoch={epoch} loss={row['loss']:.5g} "
                f"l_hom={l_hom_hard:.5g} best={best_l_hom:.5g}@{best_epoch}",
                flush=True,
            )
            if epoch >= int(min_epochs) and stale >= int(patience):
                print(f"[vgae-s2] early stop at epoch={epoch}", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out_f = model(x, edge_index, edge_weight)
            hash_scores = soft_role_probs(out_f["role_logits"]).cpu().numpy()
        rids, region_scores = pool_hash_scores_to_regions(hash_scores, incidence)
        labels = size_constrained_assign(region_scores, ratios=ratios, seed=seed)
        hard = compute_l_hom(labels, groups, soft=False)
        rows = assignment_rows(rids, labels, fold_prefix="vgae_hash")
        split_csv = assignment_rows_to_split_csv(rows, out_dir)
        meta = {
            "stage": 2,
            "grain": "hash",
            "device": device,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_l_hom": best_l_hom,
            "final_l_hom": float(hard["l_hom"]),
            "final_mean_sd_ortho": float(hard["mean_sd_ortho"]),
            "final_mean_sd_para": float(hard["mean_sd_para"]),
            "n_hash_nodes": pack.n_nodes,
            "n_regions": len(rids),
            "k": k,
            "homology_in_encoder": False,
            "split_csv": str(split_csv),
            "counts": {r: labels.count(r) for r in ("train", "test", "val")},
        }
        (out_dir / "train_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        close_dual(writer, tb_logger)
        _append_status(queue_name, "COMPLETED", note=f"best_l_hom={best_l_hom:.6g}")
        return meta
    except Exception as exc:
        _append_status(queue_name, "FAILED", note=str(exc)[:500])
        raise


# Silence unused import warning for re-export convenience
_ = run_vgae_train
