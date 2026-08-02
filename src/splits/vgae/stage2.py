"""Stage-2 hash-graph VGAE train → region-level split.csv.

Supports legacy objective (default) and additive ``loss_mode=homology_first``.
"""
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
from src.splits.vgae.homology_loss import (
    EmaTermNorm,
    compute_l_hom,
    load_homology_groups,
    write_homology_sidecar,
)
from src.splits.vgae.model import (
    ClassicVGAE,
    gumbel_softmax_roles,
    gumbel_tau_schedule,
    soft_role_probs,
)
from src.splits.vgae.train import (
    HOMOLOGY_FIRST_DEFAULTS,
    _append_status,
    _write_epoch_logs,
    compose_objective,
    random_split_l_hom_baseline,
    resolve_device,
    run_vgae_train,
)
from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry
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
    loss_mode: str = "legacy",
    max_gpu_used_mib: float = 512.0,
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
            project_dim=train_kw.get("project_dim"),
            project_seed=int(train_kw.get("project_seed", seed)),
        )
    assert_no_homology_features(pack.feature_names)
    # Make real-data scale obvious: GCN is on 4**k hash nodes; L_hom/split on all regions
    _n_r = len(incidence["region_ids"])
    _nnz = int(np.asarray(incidence["indices"]).shape[0])
    _expected_vocab = 4 ** int(k)
    print(
        f"[vgae-s2] real panel: n_regions={_n_r} marked={marked_dir} "
        f"n_hash_nodes={pack.n_nodes} (k={k}, 4**k={_expected_vocab}) "
        f"n_edges={pack.n_edges} incidence_nnz={_nnz} "
        f"max_ids={max_ids!r} homology_in_encoder=False",
        flush=True,
    )
    if max_ids is not None:
        print(
            f"[vgae-s2] WARNING: max_ids={max_ids} truncates the region panel "
            "(not a full-panel run)",
            flush=True,
        )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    mode = str(loss_mode).lower().strip()
    hf = HOMOLOGY_FIRST_DEFAULTS
    beta_kl = float(train_kw.get("beta_kl", 1.0))
    lambda_hom = float(train_kw.get("lambda_hom", 1.0))
    lambda_size = float(train_kw.get("lambda_size", 1.0))
    alpha_recon = float(train_kw.get("alpha_recon", hf["alpha_recon"]))
    beta_kl_max = float(train_kw.get("beta_kl_max", hf["beta_kl_max"]))
    kl_anneal_epochs = int(train_kw.get("kl_anneal_epochs", hf["kl_anneal_epochs"]))
    gumbel_tau_start = float(train_kw.get("gumbel_tau_start", hf["gumbel_tau_start"]))
    gumbel_tau_end = float(train_kw.get("gumbel_tau_end", hf["gumbel_tau_end"]))
    gumbel_anneal_epochs = int(
        train_kw.get("gumbel_anneal_epochs", hf["gumbel_anneal_epochs"])
    )
    lambda_para = float(train_kw.get("lambda_para", hf["lambda_para"]))
    lambda_ortho = float(train_kw.get("lambda_ortho", hf["lambda_ortho"]))
    ema_decay = float(train_kw.get("ema_decay", hf["ema_decay"]))
    hom_max_groups = int(train_kw.get("hom_max_groups", hf["hom_max_groups"]))
    if mode == "homology_first":
        if float(lambda_hom) == 1.0:
            lambda_hom = float(hf["lambda_hom"])
        if float(lambda_size) == 1.0:
            lambda_size = float(hf["lambda_size"])

    device, gpu_idx = resolve_device(
        device,
        peak_ram_gib=float(peak_ram_gib),
        wait_poll_sec=float(wait_poll_sec),
        label=f"vgae_stage2:{out_dir.name}",
        max_used_mib=float(max_gpu_used_mib),
        register_waiter=True,
    )

    queue_name = f"vgae_{out_dir.name}"
    append_queue_entry(
        queue_name,
        job=f"stage2_hash_vgae out={out_dir}",
        pid=os.getpid(),
        estimated_time="2-6h",
        job_class=CLASS_GPU_TRAIN,
        peak_ram_gib=peak_ram_gib,
        gpus=(gpu_idx,) if gpu_idx is not None else (),
        log=str(out_dir / "logs" / "metrics.log"),
        resources=f"device={device} n_hash={pack.n_nodes} loss_mode={mode}",
    )

    try:
        region_id_list = [str(x) for x in incidence["region_ids"]]
        groups = load_homology_groups(region_id_list, homology_table)
        write_homology_sidecar(
            out_dir / "pack" / "node_homology_regions.tsv", region_id_list, groups
        )

        baseline = random_split_l_hom_baseline(
            len(region_id_list), groups, ratios=ratios, seed=int(seed)
        )
        (out_dir / "random_split_baseline.json").write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[vgae-s2] random-split baseline L_hom={baseline['l_hom']:.6g}",
            flush=True,
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
        target_frac = torch.as_tensor(
            role_target_fractions(ratios), dtype=torch.float32, device=dev
        )
        ema = EmaTermNorm(decay=float(ema_decay)) if mode == "homology_first" else None

        writer = open_summary_writer(out_dir)
        tb_logger = open_tensorboard_logger(out_dir)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_scalar_pair(
            writer, tb_logger, "baseline/random_l_hom", baseline["l_hom"], 0
        )

        n_r = len(region_id_list)
        indptr_np = np.asarray(incidence["indptr"], dtype=np.int32)
        indices_np = np.asarray(incidence["indices"], dtype=np.int32)
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
            tau = gumbel_tau_schedule(
                epoch,
                tau_start=gumbel_tau_start,
                tau_end=gumbel_tau_end,
                t_anneal=gumbel_anneal_epochs,
            )
            if mode == "homology_first":
                soft_h = gumbel_softmax_roles(out["role_logits"], tau=tau, hard=False)
                soft_log = soft_role_probs(out["role_logits"])
            else:
                soft_h = soft_role_probs(out["role_logits"])
                soft_log = soft_h

            pooled = torch.zeros((n_r, 3), device=dev, dtype=soft_h.dtype)
            counts = torch.zeros((n_r, 1), device=dev, dtype=soft_h.dtype)
            if hash_idx_t.numel() > 0:
                pooled.index_add_(0, region_idx_t, soft_h.index_select(0, hash_idx_t))
                counts.index_add_(0, region_idx_t, ones_inc.to(dtype=soft_h.dtype))
            pooled = pooled / counts.clamp_min(1.0)
            empty = counts.squeeze(-1) <= 0
            if empty.any():
                pooled = pooled.clone()
                pooled[empty] = soft_h.new_tensor([1.0, 0.0, 0.0])
            pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # Softmax pool for size when homology_first (stable fractions)
            if mode == "homology_first":
                pooled_size = torch.zeros((n_r, 3), device=dev, dtype=soft_log.dtype)
                counts_s = torch.zeros((n_r, 1), device=dev, dtype=soft_log.dtype)
                if hash_idx_t.numel() > 0:
                    pooled_size.index_add_(
                        0, region_idx_t, soft_log.index_select(0, hash_idx_t)
                    )
                    counts_s.index_add_(
                        0, region_idx_t, ones_inc.to(dtype=soft_log.dtype)
                    )
                pooled_size = pooled_size / counts_s.clamp_min(1.0)
                empty_s = counts_s.squeeze(-1) <= 0
                if empty_s.any():
                    pooled_size = pooled_size.clone()
                    pooled_size[empty_s] = soft_log.new_tensor([1.0, 0.0, 0.0])
                pooled_size = pooled_size / pooled_size.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
            else:
                pooled_size = pooled

            recon = ClassicVGAE.recon_loss_neg_sample(out["z"], edge_index, edge_weight)
            kl = ClassicVGAE.kl_loss(out["mu"], out["logstd"])
            if mode == "homology_first":
                hom = compute_l_hom(
                    pooled,
                    groups,
                    soft=True,
                    weighted=True,
                    max_groups=hom_max_groups,
                    subset_seed=int(seed) + int(epoch),
                    lambda_para=lambda_para,
                    lambda_ortho=lambda_ortho,
                )
            else:
                hom = compute_l_hom(pooled, groups, soft=True)
            sz = size_loss(pooled_size, target_frac)
            composed = compose_objective(
                recon=recon,
                kl=kl,
                l_hom=hom["l_hom"],
                size=sz,
                loss_mode=mode,
                epoch=epoch,
                beta_kl=beta_kl,
                lambda_hom=lambda_hom,
                lambda_size=lambda_size,
                alpha_recon=alpha_recon,
                beta_kl_max=beta_kl_max,
                kl_anneal_epochs=kl_anneal_epochs,
                ema=ema,
            )
            loss = composed["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                out_e = model(x, edge_index, edge_weight)
                hash_scores = soft_role_probs(out_e["role_logits"]).cpu().numpy()
            rids, region_scores = pool_hash_scores_to_regions(hash_scores, incidence)
            labels = size_constrained_assign(
                region_scores, ratios=ratios, seed=seed + epoch
            )
            hard = compute_l_hom(labels, groups, soft=False)
            l_hom_hard = float(hard["l_hom"])

            mag_r = abs(float(composed["term_recon"]))
            mag_k = abs(float(composed["term_kl"]))
            mag_h = abs(float(composed["term_hom"]))
            mag_s = abs(float(composed["term_size"]))
            mag_sum = mag_r + mag_k + mag_h + mag_s + 1e-12

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
                "recon_norm": composed["recon_norm"],
                "kl_norm": composed["kl_norm"],
                "beta_kl": composed["beta_used"],
                "gumbel_tau": float(tau) if mode == "homology_first" else None,
                "hom_grad_share": float(mag_h / mag_sum),
                "ema_recon": composed["ema_recon"],
                "ema_kl": composed["ema_kl"],
            }
            epoch_rows.append(row)
            _write_epoch_logs(out_dir, epoch_rows)
            log_scalar_pair(writer, tb_logger, "train/loss", row["loss"], epoch)
            log_scalar_pair(writer, tb_logger, "train/l_hom_soft", row["l_hom_soft"], epoch)
            if row["recon_norm"] is not None:
                log_scalar_pair(
                    writer, tb_logger, "train/recon_norm", row["recon_norm"], epoch
                )
                log_scalar_pair(writer, tb_logger, "train/kl_norm", row["kl_norm"], epoch)
                log_scalar_pair(
                    writer, tb_logger, "train/hom_grad_share", row["hom_grad_share"], epoch
                )
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
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                torch.save(
                    {"epoch": epoch, "model": best_state, "l_hom": best_l_hom},
                    ckpt_dir / "best.pt",
                )
                stale = 0
            elif epoch >= int(min_epochs):
                stale += 1

            print(
                f"[vgae-s2] epoch={epoch} loss={row['loss']:.5g} "
                f"l_hom={l_hom_hard:.5g} best={best_l_hom:.5g}@{best_epoch} "
                f"mode={mode}",
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
            "loss_mode": mode,
            "best_epoch": best_epoch,
            "best_l_hom": best_l_hom,
            "final_l_hom": float(hard["l_hom"]),
            "final_mean_sd_ortho": float(hard["mean_sd_ortho"]),
            "final_mean_sd_para": float(hard["mean_sd_para"]),
            "random_baseline_l_hom": baseline["l_hom"],
            "n_hash_nodes": pack.n_nodes,
            "n_regions": len(rids),
            "k": k,
            "lambda_hom": lambda_hom,
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
