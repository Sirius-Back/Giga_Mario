"""VGAE training loop: recon + KL + size + L_hom; early stop after ≥25 epochs."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.pipeline.job_queue import (
    CLASS_GPU_TRAIN,
    append_queue_entry,
    wait_until_launchable,
)
from src.splits.vgae.assign import (
    ROLE_ORDER,
    assignment_rows,
    role_target_fractions,
    size_constrained_assign,
    size_loss,
)
from src.splits.vgae.graph_data import PackedGraph, assert_no_homology_features, load_packed_graph
from src.splits.vgae.homology_loss import (
    compute_l_hom,
    load_homology_groups,
    write_homology_sidecar,
)
from src.splits.vgae.model import ClassicVGAE, soft_role_probs
from src.splits.sbs.assign import assignment_rows_to_split_csv
from src.tb_logging import close_dual, log_scalar_pair, open_summary_writer, open_tensorboard_logger


def _pick_free_gpu() -> int:
    """Return a GPU index with lowest memory used (does not kill processes)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for VGAE train (plan: 1×GPU)")
    best_i = 0
    best_used = None
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
        except Exception:
            used = 0
        if best_used is None or used < best_used:
            best_used = used
            best_i = i
    return int(best_i)


def _append_status(queue_name: str, status: str, *, note: str = "") -> None:
    """Append a status follow-up line to queue.md (do not delete history)."""
    from src.pipeline.job_queue import queue_path

    p = queue_path()
    lines = [f"\n### {queue_name} — {status}"]
    lines.append(f"- **update time:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    if note:
        lines.append(f"- **note:** {note}")
    lines.append(f"- **status:** {status}")
    with p.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_epoch_logs(
    out_dir: Path,
    epoch_rows: list[dict[str, Any]],
) -> None:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jsonl = logs / "train_metrics.jsonl"
    mlog = logs / "metrics.log"
    lines_j: list[str] = []
    lines_l: list[str] = []
    for rec in epoch_rows:
        ep = int(rec["epoch"])
        obj = {
            "epoch": ep,
            "train": {
                "loss": float(rec["loss"]),
                "recon": float(rec["recon"]),
                "kl": float(rec["kl"]),
                "l_hom_soft": float(rec["l_hom_soft"]),
                "size": float(rec["size"]),
            },
            "validation": {
                "loss": float(rec["l_hom_hard"]),
                "l_hom": float(rec["l_hom_hard"]),
                "mean_sd_ortho": float(rec["mean_sd_ortho"]),
                "mean_sd_para": float(rec["mean_sd_para"]),
            },
            "test": {
                "loss": float(rec["l_hom_hard"]),
                "l_hom": float(rec["l_hom_hard"]),
            },
        }
        lines_j.append(json.dumps(obj, sort_keys=True))
        lines_l.append(
            f"epoch={ep} train_loss={obj['train']['loss']:.6g} "
            f"val_l_hom={obj['validation']['l_hom']:.6g} "
            f"sd_ortho={obj['validation']['mean_sd_ortho']:.6g} "
            f"sd_para={obj['validation']['mean_sd_para']:.6g}"
        )
        ep_dir = logs / f"epoch{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "metrics.json").write_text(
            json.dumps(obj, indent=2) + "\n", encoding="utf-8"
        )
    jsonl.write_text("\n".join(lines_j) + ("\n" if lines_j else ""), encoding="utf-8")
    mlog.write_text("\n".join(lines_l) + ("\n" if lines_l else ""), encoding="utf-8")


def run_vgae_train(
    pack: PackedGraph | Path,
    out_dir: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    hidden_dim: int = 64,
    latent_dim: int = 32,
    lr: float = 1e-3,
    beta_kl: float = 1.0,
    lambda_hom: float = 1.0,
    lambda_size: float = 1.0,
    max_epochs: int = 200,
    min_epochs: int = 25,
    patience: int = 10,
    device: str | None = None,
    homology_table: Path | None = None,
    peak_ram_gib: float = 12.0,
    wait_poll_sec: float = 600.0,
    register_queue: bool = True,
) -> dict[str, Any]:
    """Train classic VGAE and write ``split.csv`` + logs under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pack, (str, Path)):
        pack = load_packed_graph(Path(pack))
    assert_no_homology_features(pack.feature_names)
    if pack.meta.get("homology_in_encoder"):
        raise ValueError("pack meta claims homology_in_encoder=True — refused")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    gpu_idx = None
    if device is None:
        gpu_idx = _pick_free_gpu()
        # Wait until this GPU is launchable under queue politics
        wait_until_launchable(
            peak_ram_gib=float(peak_ram_gib),
            gpus=(gpu_idx,),
            job_class=CLASS_GPU_TRAIN,
            timeout_sec=6 * 3600,
            poll_sec=float(wait_poll_sec),
            label=f"vgae_train:{out_dir.name}",
        )
        # Re-pick after wait (another job may have freed a better GPU)
        gpu_idx = _pick_free_gpu()
        wait_until_launchable(
            peak_ram_gib=float(peak_ram_gib),
            gpus=(gpu_idx,),
            job_class=CLASS_GPU_TRAIN,
            timeout_sec=6 * 3600,
            poll_sec=float(wait_poll_sec),
            label=f"vgae_train_confirm:{out_dir.name}",
        )
        device = f"cuda:{gpu_idx}"
    else:
        if device.startswith("cuda"):
            parts = device.split(":")
            gpu_idx = int(parts[1]) if len(parts) > 1 else 0
            wait_until_launchable(
                peak_ram_gib=float(peak_ram_gib),
                gpus=(gpu_idx,),
                job_class=CLASS_GPU_TRAIN,
                timeout_sec=6 * 3600,
                poll_sec=float(wait_poll_sec),
                label=f"vgae_train:{out_dir.name}",
            )

    queue_name = f"vgae_{out_dir.name}"
    if register_queue:
        append_queue_entry(
            queue_name,
            job=f"python -m src.splits.vgae --out {out_dir}",
            pid=os.getpid(),
            estimated_time="2-6h",
            job_class=CLASS_GPU_TRAIN,
            peak_ram_gib=float(peak_ram_gib),
            gpus=(gpu_idx,) if gpu_idx is not None else (),
            log=str(out_dir / "logs" / "metrics.log"),
            resources=f"device={device} n={pack.n_nodes} e={pack.n_edges}",
        )

    try:
        groups = load_homology_groups(pack.ids, homology_table)
        write_homology_sidecar(out_dir / "pack" / "node_homology.tsv", pack.ids, groups)
        # Ensure pack dir also has a copy of feature meta flag
        pack_out = out_dir / "pack"
        pack_out.mkdir(parents=True, exist_ok=True)
        if pack.pack_dir.resolve() != pack_out.resolve():
            # Copy essential pack files into run tree
            import shutil

            for name in (
                "node_features.npz",
                "edges_weighted.npz",
                "ids.txt",
                "feature_meta.json",
            ):
                src = pack.pack_dir / name
                if src.is_file():
                    shutil.copy2(src, pack_out / name)

        dev = torch.device(device)
        x = torch.as_tensor(pack.x, dtype=torch.float32, device=dev)
        # Firewall: feature dim must match compositional names only
        assert_no_homology_features(pack.feature_names)
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
            n_roles=3,
        ).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        target_frac = torch.as_tensor(
            role_target_fractions(ratios), dtype=torch.float32, device=dev
        )

        writer = open_summary_writer(out_dir)
        tb_logger = open_tensorboard_logger(out_dir)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_l_hom = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = -1
        stale = 0
        epoch_rows: list[dict[str, Any]] = []

        for epoch in range(1, int(max_epochs) + 1):
            model.train()
            out = model(x, edge_index, edge_weight)
            # Re-assert no homology dim creep
            if out["z"].size(0) != x.size(0):
                raise RuntimeError("latent/node count mismatch")
            soft = soft_role_probs(out["role_logits"])
            recon = ClassicVGAE.recon_loss_neg_sample(
                out["z"], edge_index, edge_weight
            )
            kl = ClassicVGAE.kl_loss(out["mu"], out["logstd"])
            hom = compute_l_hom(soft, groups, soft=True)
            l_hom_soft = hom["l_hom"]
            sz = size_loss(soft, target_frac)
            loss = (
                recon
                + float(beta_kl) * kl
                + float(lambda_hom) * l_hom_soft
                + float(lambda_size) * sz
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # Hard eval
            model.eval()
            with torch.no_grad():
                out_e = model(x, edge_index, edge_weight)
                scores = soft_role_probs(out_e["role_logits"]).cpu().numpy()
            labels = size_constrained_assign(scores, ratios=ratios, seed=seed + epoch)
            hard = compute_l_hom(labels, groups, soft=False)
            l_hom_hard = float(hard["l_hom"])
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "recon": float(recon.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "l_hom_soft": float(l_hom_soft.detach().cpu()),
                "size": float(sz.detach().cpu()),
                "l_hom_hard": l_hom_hard,
                "mean_sd_ortho": float(hard["mean_sd_ortho"]),
                "mean_sd_para": float(hard["mean_sd_para"]),
            }
            epoch_rows.append(row)
            _write_epoch_logs(out_dir, epoch_rows)

            log_scalar_pair(writer, tb_logger, "train/loss", row["loss"], epoch)
            log_scalar_pair(writer, tb_logger, "train/recon", row["recon"], epoch)
            log_scalar_pair(writer, tb_logger, "train/kl", row["kl"], epoch)
            log_scalar_pair(writer, tb_logger, "train/l_hom_soft", row["l_hom_soft"], epoch)
            log_scalar_pair(writer, tb_logger, "validation/l_hom", l_hom_hard, epoch)
            log_scalar_pair(
                writer, tb_logger, "validation/mean_sd_ortho", row["mean_sd_ortho"], epoch
            )
            log_scalar_pair(
                writer, tb_logger, "validation/mean_sd_para", row["mean_sd_para"], epoch
            )

            improved = l_hom_hard < best_l_hom - 1e-6
            if improved:
                best_l_hom = l_hom_hard
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {"epoch": epoch, "model": best_state, "l_hom": best_l_hom},
                    ckpt_dir / "best.pt",
                )
                stale = 0
            else:
                if epoch >= int(min_epochs):
                    stale += 1

            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "l_hom": l_hom_hard},
                ckpt_dir / "last.pt",
            )

            print(
                f"[vgae] epoch={epoch} loss={row['loss']:.5g} "
                f"l_hom_hard={l_hom_hard:.5g} "
                f"sd_o={row['mean_sd_ortho']:.5g} sd_p={row['mean_sd_para']:.5g} "
                f"best={best_l_hom:.5g}@{best_epoch} stale={stale}",
                flush=True,
            )

            if epoch >= int(min_epochs) and stale >= int(patience):
                print(
                    f"[vgae] early stop at epoch={epoch} "
                    f"(min_epochs={min_epochs}, patience={patience})",
                    flush=True,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out_f = model(x, edge_index, edge_weight)
            scores = soft_role_probs(out_f["role_logits"]).cpu().numpy()
            z = out_f["z"].cpu().numpy()
        labels = size_constrained_assign(scores, ratios=ratios, seed=seed)
        hard = compute_l_hom(labels, groups, soft=False)
        rows = assignment_rows(pack.ids, labels, fold_prefix="vgae")
        split_csv = assignment_rows_to_split_csv(rows, out_dir)
        np.savez_compressed(out_dir / "latents.npz", z=z, scores=scores)
        counts = {r: labels.count(r) for r in ROLE_ORDER}
        meta = {
            "device": device,
            "seed": seed,
            "ratios": list(ratios),
            "best_epoch": best_epoch,
            "best_l_hom": best_l_hom,
            "final_l_hom": float(hard["l_hom"]),
            "final_mean_sd_ortho": float(hard["mean_sd_ortho"]),
            "final_mean_sd_para": float(hard["mean_sd_para"]),
            "counts": counts,
            "n_nodes": pack.n_nodes,
            "n_edges": pack.n_edges,
            "k": pack.k,
            "min_epochs": min_epochs,
            "patience": patience,
            "max_epochs": max_epochs,
            "homology_in_encoder": False,
            "split_csv": str(split_csv),
        }
        (out_dir / "train_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        close_dual(writer, tb_logger)
        if register_queue:
            _append_status(queue_name, "COMPLETED", note=f"best_l_hom={best_l_hom:.6g}")
        return meta
    except Exception as exc:
        if register_queue:
            _append_status(queue_name, "FAILED", note=str(exc)[:500])
        raise
