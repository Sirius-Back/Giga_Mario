"""MLP-VAE train: homology_first primary + legacy loss logged; early stop ≥25."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.pipeline.job_queue import (
    CLASS_CPU_RAM_HEAVY,
    CLASS_GPU_TRAIN,
    CLASS_WAITER,
    append_queue_entry,
    wait_until_launchable,
)
from src.splits.sbs.assign import assignment_rows_to_split_csv
from src.splits.vae.features import PackedFeatures, load_packed_features, pack_feature_table
from src.splits.vae.model import MlpVAE
from src.splits.vgae.assign import (
    ROLE_ORDER,
    assignment_rows,
    role_target_fractions,
    size_constrained_assign,
    size_loss,
)
from src.splits.vgae.graph_data import assert_no_homology_features
from src.splits.vgae.homology_loss import (
    EmaTermNorm,
    compute_l_hom,
    load_homology_groups,
    write_homology_sidecar,
)
from src.splits.vgae.model import (
    gumbel_softmax_roles,
    gumbel_tau_schedule,
    kl_beta_schedule,
    soft_role_probs,
)
from src.splits.vgae.train import (
    HOMOLOGY_FIRST_DEFAULTS,
    _append_status,
    compose_objective,
    random_split_l_hom_baseline,
    resolve_device,
    wait_for_free_gpu,
)
from src.tb_logging import close_dual, log_scalar_pair, open_summary_writer, open_tensorboard_logger


def _write_epoch_logs(out_dir: Path, epoch_rows: list[dict[str, Any]]) -> None:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jsonl = logs / "train_metrics.jsonl"
    mlog = logs / "metrics.log"
    lines_j: list[str] = []
    lines_l: list[str] = []
    for rec in epoch_rows:
        ep = int(rec["epoch"])
        train_block = {
            "loss": float(rec["loss"]),
            "loss_legacy": float(rec["loss_legacy"]),
            "recon": float(rec["recon"]),
            "kl": float(rec["kl"]),
            "l_hom_soft": float(rec["l_hom_soft"]),
            "l_hom_soft_legacy": float(rec["l_hom_soft_legacy"]),
            "size": float(rec["size"]),
        }
        for key in (
            "recon_norm",
            "kl_norm",
            "beta_kl",
            "gumbel_tau",
            "hom_grad_share",
            "ema_recon",
            "ema_kl",
        ):
            if key in rec and rec[key] is not None:
                train_block[key] = float(rec[key])
        obj = {
            "epoch": ep,
            "train": train_block,
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
            f"legacy={obj['train']['loss_legacy']:.6g} "
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


def resolve_vae_device(
    device: str | None,
    *,
    peak_ram_gib: float,
    wait_poll_sec: float,
    label: str,
    prefer_cpu: bool = True,
    max_gpu_used_mib: float = 512.0,
    register_waiter: bool = True,
) -> tuple[str, int | None, str]:
    """Prefer CPU when RAM headroom fits; else free GPU. Never kill PIDs."""
    if device is not None:
        if device == "cpu":
            wait_until_launchable(
                peak_ram_gib=float(peak_ram_gib),
                gpus=(),
                job_class=CLASS_CPU_RAM_HEAVY,
                timeout_sec=6 * 3600,
                poll_sec=float(wait_poll_sec),
                label=label,
            )
            return "cpu", None, CLASS_CPU_RAM_HEAVY
        # Explicit CUDA — reuse VGAE free-GPU waiter
        dev, gpu_idx = resolve_device(
            device,
            peak_ram_gib=float(peak_ram_gib),
            wait_poll_sec=float(wait_poll_sec),
            label=label,
            max_used_mib=float(max_gpu_used_mib),
            register_waiter=register_waiter,
        )
        return dev, gpu_idx, CLASS_GPU_TRAIN

    if prefer_cpu:
        waiter_name = f"waiter_{label}"
        if register_waiter:
            append_queue_entry(
                waiter_name,
                job=f"wait_ram:{label}",
                pid=os.getpid(),
                estimated_time=f"poll {wait_poll_sec:.0f}s",
                status="RUNNING",
                job_class=CLASS_WAITER,
                peak_ram_gib=0.0,
                gpus=(),
            )
        try:
            wait_until_launchable(
                peak_ram_gib=float(peak_ram_gib),
                gpus=(),
                job_class=CLASS_CPU_RAM_HEAVY,
                timeout_sec=6 * 3600,
                poll_sec=float(wait_poll_sec),
                label=label,
            )
            if register_waiter:
                _append_status(waiter_name, "COMPLETED", note="CPU RAM headroom OK")
            print(f"[vae] {label}: launching on CPU", flush=True)
            return "cpu", None, CLASS_CPU_RAM_HEAVY
        except TimeoutError:
            if register_waiter:
                _append_status(
                    waiter_name, "COMPLETED", note="CPU wait timed out → try free GPU"
                )
            print(f"[vae] {label}: CPU wait timed out; trying free GPU…", flush=True)

    gpu_idx = wait_for_free_gpu(
        peak_ram_gib=float(peak_ram_gib),
        max_used_mib=float(max_gpu_used_mib),
        poll_sec=float(wait_poll_sec),
        label=f"{label}_gpu",
        register_waiter=register_waiter,
    )
    return f"cuda:{gpu_idx}", gpu_idx, CLASS_GPU_TRAIN


def run_vae_train(
    pack: PackedFeatures | Path,
    out_dir: Path,
    *,
    features_path: Path | None = None,
    k: int = 4,
    seed: int = 42,
    ratios: tuple[float, float, float] = (3.0, 1.0, 1.0),
    hidden_dim: int = 256,
    latent_dim: int = 64,
    lr: float = 1e-3,
    max_epochs: int = 200,
    min_epochs: int = 25,
    patience: int = 10,
    device: str | None = None,
    prefer_cpu: bool = True,
    homology_table: Path | None = None,
    peak_ram_gib: float = 8.0,
    wait_poll_sec: float = 600.0,
    register_queue: bool = True,
    lambda_hom: float | None = None,
    lambda_size: float | None = None,
    lambda_para: float | None = None,
    lambda_ortho: float | None = None,
    alpha_recon: float | None = None,
    beta_kl_max: float | None = None,
    source_label: str | None = None,
    project_dim: int | None = None,
    project_seed: int = 42,
    keep_memmap: bool = False,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Train MLP-VAE with homology_first (legacy logged); write ``split.csv``.

    For full k=7 16384-d: ``keep_memmap=True``, ``project_dim=None``,
    ``batch_size`` (e.g. 2048) on GPU — X stays memory-mapped on disk.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = out_dir / "pack"

    pack_kw = dict(
        k=k,
        source_label=source_label,
        project_dim=project_dim,
        project_seed=int(project_seed),
        keep_memmap=bool(keep_memmap),
    )

    if (pack_dir / "feature_meta.json").is_file():
        pack = load_packed_features(pack_dir)
    elif isinstance(pack, PackedFeatures):
        if pack.pack_dir.resolve() != pack_dir.resolve():
            src = features_path or pack.meta.get("source_features")
            if not src:
                raise ValueError("PackedFeatures has no source_features; pass features_path")
            pack = pack_feature_table(Path(src), pack_dir, **pack_kw)
    elif features_path is not None:
        pack = pack_feature_table(Path(features_path), pack_dir, **pack_kw)
    else:
        pack = pack_feature_table(Path(pack), pack_dir, **pack_kw)

    assert_no_homology_features(pack.feature_names)
    native_dim = 4 ** int(k)
    proj = pack.meta.get("feature_projection") or {}
    if pack.x.shape[1] != native_dim and not proj.get("applied"):
        raise ValueError(
            f"expected {native_dim} features for k={k} (or projected pack); "
            f"got {pack.x.shape[1]}"
        )
    if project_dim is not None and pack.x.shape[1] not in (native_dim, int(project_dim)):
        # Allow reuse of already-projected pack
        if not proj.get("applied"):
            raise ValueError(
                f"project_dim={project_dim} but pack has {pack.x.shape[1]} features"
            )

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    hf = HOMOLOGY_FIRST_DEFAULTS
    alpha_recon = float(alpha_recon if alpha_recon is not None else hf["alpha_recon"])
    beta_kl_max = float(beta_kl_max if beta_kl_max is not None else hf["beta_kl_max"])
    kl_anneal_epochs = int(hf["kl_anneal_epochs"])
    gumbel_tau_start = float(hf["gumbel_tau_start"])
    gumbel_tau_end = float(hf["gumbel_tau_end"])
    gumbel_anneal_epochs = int(hf["gumbel_anneal_epochs"])
    lambda_para = float(lambda_para if lambda_para is not None else hf["lambda_para"])
    lambda_ortho = float(lambda_ortho if lambda_ortho is not None else hf["lambda_ortho"])
    ema_decay = float(hf["ema_decay"])
    hom_max_groups = int(hf["hom_max_groups"])
    lambda_hom = float(
        lambda_hom if lambda_hom is not None else hf["lambda_hom"]
    )
    lambda_size = float(
        lambda_size if lambda_size is not None else hf["lambda_size"]
    )

    device_s, gpu_idx, job_class = resolve_vae_device(
        device,
        peak_ram_gib=float(peak_ram_gib),
        wait_poll_sec=float(wait_poll_sec),
        label=f"vae_train:{out_dir.name}",
        prefer_cpu=prefer_cpu,
        register_waiter=register_queue,
    )

    queue_name = f"vae_{out_dir.name}"
    if register_queue:
        append_queue_entry(
            queue_name,
            job=f"python -m src.splits.vae --out {out_dir}",
            pid=os.getpid(),
            estimated_time="1-4h",
            job_class=job_class,
            peak_ram_gib=float(peak_ram_gib),
            gpus=(gpu_idx,) if gpu_idx is not None else (),
            log=str(out_dir / "logs" / "metrics.log"),
            resources=f"device={device_s} n={pack.n_nodes} k={k} loss=homology_first",
        )

    try:
        groups = load_homology_groups(pack.ids, homology_table)
        write_homology_sidecar(pack_dir / "node_homology.tsv", pack.ids, groups)

        baseline = random_split_l_hom_baseline(
            pack.n_nodes, groups, ratios=ratios, seed=int(seed)
        )
        (out_dir / "random_split_baseline.json").write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[vae] random-split baseline L_hom={baseline['l_hom']:.6g} (seed={seed})",
            flush=True,
        )

        dev = torch.device(device_s)
        n = int(pack.n_nodes)
        in_dim = int(pack.x.shape[1])
        use_batches = batch_size is not None and int(batch_size) > 0 and int(batch_size) < n
        bs = int(batch_size) if use_batches else n
        x_np = pack.x  # may be memmap
        # Full-tensor path only when it fits (small packs)
        x_full: torch.Tensor | None = None
        if not use_batches:
            x_full = torch.as_tensor(np.asarray(x_np, dtype=np.float32), device=dev)

        model = MlpVAE(
            in_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_roles=3,
        ).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        target_frac = torch.as_tensor(
            role_target_fractions(ratios), dtype=torch.float32, device=dev
        )
        ema = EmaTermNorm(decay=float(ema_decay))

        writer = open_summary_writer(out_dir)
        tb_logger = open_tensorboard_logger(out_dir)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_scalar_pair(
            writer, tb_logger, "baseline/random_l_hom", baseline["l_hom"], 0
        )
        print(
            f"[vae] n={n} in_dim={in_dim} batch_size={bs if use_batches else 'full'} "
            f"device={device_s} storage={pack.meta.get('storage')}",
            flush=True,
        )

        def _batch_indices() -> list[np.ndarray]:
            if not use_batches:
                return [np.arange(n, dtype=np.int64)]
            idxs = []
            for i0 in range(0, n, bs):
                idxs.append(np.arange(i0, min(n, i0 + bs), dtype=np.int64))
            return idxs

        def _x_batch(idx: np.ndarray) -> torch.Tensor:
            if x_full is not None:
                return x_full.index_select(0, torch.as_tensor(idx, device=dev))
            # memmap / numpy → GPU microbatch
            return torch.as_tensor(np.asarray(x_np[idx], dtype=np.float32), device=dev)

        best_l_hom = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = -1
        stale = 0
        epoch_rows: list[dict[str, Any]] = []

        for epoch in range(1, int(max_epochs) + 1):
            model.train()
            tau = gumbel_tau_schedule(
                epoch,
                tau_start=gumbel_tau_start,
                tau_end=gumbel_tau_end,
                t_anneal=gumbel_anneal_epochs,
            )
            batches = _batch_indices()
            opt.zero_grad(set_to_none=True)
            beta_t = kl_beta_schedule(
                int(epoch), beta_max=float(beta_kl_max), t_anneal=int(kl_anneal_epochs)
            )

            # Phase 1: EMA-scaled recon+KL microbatches (grad accumulate)
            recon_acc = 0.0
            kl_acc = 0.0
            n_seen = 0
            for idx in batches:
                xb = _x_batch(idx)
                out_b = model(xb)
                recon_b = MlpVAE.recon_loss_mse(xb, out_b["x_hat"])
                kl_b = MlpVAE.kl_loss(out_b["mu"], out_b["logstd"])
                w = float(len(idx)) / float(n)
                er = (
                    float(ema.ema_recon)
                    if ema.ema_recon is not None
                    else max(float(recon_b.detach().cpu()), 1e-6)
                )
                ek = (
                    float(ema.ema_kl)
                    if ema.ema_kl is not None
                    else max(float(kl_b.detach().cpu()), 1e-6)
                )
                (
                    (float(alpha_recon) * recon_b / er + float(beta_t) * kl_b / ek) * w
                ).backward()
                recon_acc += float(recon_b.detach().cpu()) * len(idx)
                kl_acc += float(kl_b.detach().cpu()) * len(idx)
                n_seen += len(idx)
            recon_mean = recon_acc / max(n_seen, 1)
            kl_mean = kl_acc / max(n_seen, 1)
            recon_t = torch.tensor(recon_mean, device=dev, dtype=torch.float32)
            kl_t = torch.tensor(kl_mean, device=dev, dtype=torch.float32)

            # Phase 2: role logits via activation checkpointing.
            # Pass CPU tensors into checkpoint so saved inputs live in host RAM
            # (full panel ≈ n×d×4), while VRAM peak stays ≈ 1 microbatch.
            logit_parts: list[torch.Tensor] = []

            def _role_logits_from_cpu(
                x_cpu: torch.Tensor, _model: nn.Module = model
            ) -> torch.Tensor:
                return _model(x_cpu.to(dev, non_blocking=True))["role_logits"]

            for idx in batches:
                x_cpu = torch.as_tensor(np.asarray(x_np[idx], dtype=np.float32))
                if use_batches:
                    logits_b = torch.utils.checkpoint.checkpoint(
                        _role_logits_from_cpu, x_cpu, use_reentrant=False
                    )
                else:
                    logits_b = _role_logits_from_cpu(x_cpu)
                logit_parts.append(logits_b)
            role_logits = torch.cat(logit_parts, dim=0)
            soft_gs = gumbel_softmax_roles(role_logits, tau=tau, hard=False)
            soft_sm = soft_role_probs(role_logits)
            hom = compute_l_hom(
                soft_gs,
                groups,
                soft=True,
                weighted=True,
                max_groups=hom_max_groups,
                subset_seed=int(seed) + int(epoch),
                lambda_para=lambda_para,
                lambda_ortho=lambda_ortho,
            )
            sz = size_loss(soft_sm, target_frac)
            composed = compose_objective(
                recon=recon_t,
                kl=kl_t,
                l_hom=hom["l_hom"],
                size=sz,
                loss_mode="homology_first",
                epoch=epoch,
                beta_kl=1.0,
                lambda_hom=lambda_hom,
                lambda_size=lambda_size,
                alpha_recon=alpha_recon,
                beta_kl_max=beta_kl_max,
                kl_anneal_epochs=kl_anneal_epochs,
                ema=ema,
            )
            (float(lambda_hom) * hom["l_hom"] + float(lambda_size) * sz).backward()
            opt.step()

            loss = composed["loss"]
            with torch.no_grad():
                hom_leg = compute_l_hom(soft_sm.detach(), groups, soft=True)
                loss_legacy = (
                    recon_t.detach()
                    + kl_t.detach()
                    + hom_leg["l_hom"]
                    + sz.detach()
                )
                recon = recon_t
                kl = kl_t

            model.eval()
            with torch.no_grad():
                score_parts: list[np.ndarray] = []
                for idx in batches:
                    xb = _x_batch(idx)
                    out_e = model(xb)
                    score_parts.append(
                        soft_role_probs(out_e["role_logits"]).cpu().numpy()
                    )
                scores = np.concatenate(score_parts, axis=0)
            labels = size_constrained_assign(scores, ratios=ratios, seed=seed + epoch)
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
                "loss_legacy": float(loss_legacy.detach().cpu()),
                "recon": float(recon.detach().cpu()),
                "kl": float(kl.detach().cpu()),
                "l_hom_soft": float(hom["l_hom"].detach().cpu()),
                "l_hom_soft_legacy": float(hom_leg["l_hom"].detach().cpu()),
                "size": float(sz.detach().cpu()),
                "l_hom_hard": l_hom_hard,
                "mean_sd_ortho": float(hard["mean_sd_ortho"]),
                "mean_sd_para": float(hard["mean_sd_para"]),
                "recon_norm": composed["recon_norm"],
                "kl_norm": composed["kl_norm"],
                "beta_kl": composed["beta_used"],
                "gumbel_tau": float(tau),
                "hom_grad_share": float(mag_h / mag_sum),
                "ema_recon": composed["ema_recon"],
                "ema_kl": composed["ema_kl"],
            }
            epoch_rows.append(row)
            _write_epoch_logs(out_dir, epoch_rows)

            log_scalar_pair(writer, tb_logger, "train/loss", row["loss"], epoch)
            log_scalar_pair(
                writer, tb_logger, "train/loss_legacy", row["loss_legacy"], epoch
            )
            log_scalar_pair(writer, tb_logger, "train/recon", row["recon"], epoch)
            log_scalar_pair(writer, tb_logger, "train/kl", row["kl"], epoch)
            log_scalar_pair(
                writer, tb_logger, "train/l_hom_soft", row["l_hom_soft"], epoch
            )
            log_scalar_pair(
                writer,
                tb_logger,
                "train/l_hom_soft_legacy",
                row["l_hom_soft_legacy"],
                epoch,
            )
            log_scalar_pair(
                writer, tb_logger, "train/recon_norm", float(row["recon_norm"]), epoch
            )
            log_scalar_pair(
                writer, tb_logger, "train/kl_norm", float(row["kl_norm"]), epoch
            )
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
                    k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()
                }
                torch.save(
                    {"epoch": epoch, "model": best_state, "l_hom": best_l_hom},
                    ckpt_dir / "best.pt",
                )
                stale = 0
            elif epoch >= int(min_epochs):
                stale += 1

            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "l_hom": l_hom_hard},
                ckpt_dir / "last.pt",
            )
            print(
                f"[vae] epoch={epoch} loss={row['loss']:.5g} "
                f"legacy={row['loss_legacy']:.5g} "
                f"l_hom_hard={l_hom_hard:.5g} best={best_l_hom:.5g}@{best_epoch} "
                f"stale={stale}",
                flush=True,
            )
            if epoch >= int(min_epochs) and stale >= int(patience):
                print(
                    f"[vae] early stop at epoch={epoch} "
                    f"(min_epochs={min_epochs}, patience={patience})",
                    flush=True,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            score_parts_f: list[np.ndarray] = []
            z_parts: list[np.ndarray] = []
            for idx in _batch_indices():
                xb = _x_batch(idx)
                out_f = model(xb)
                score_parts_f.append(soft_role_probs(out_f["role_logits"]).cpu().numpy())
                z_parts.append(out_f["z"].cpu().numpy())
            scores = np.concatenate(score_parts_f, axis=0)
            z = np.concatenate(z_parts, axis=0)
        labels = size_constrained_assign(scores, ratios=ratios, seed=seed)
        hard = compute_l_hom(labels, groups, soft=False)
        rows = assignment_rows(pack.ids, labels, fold_prefix="vae")
        split_csv = assignment_rows_to_split_csv(rows, out_dir)
        np.savez_compressed(out_dir / "latents.npz", z=z, scores=scores)
        counts = {r: labels.count(r) for r in ROLE_ORDER}
        meta = {
            "device": device_s,
            "seed": seed,
            "ratios": list(ratios),
            "loss_mode": "homology_first",
            "k": int(k),
            "batch_size": int(bs) if use_batches else None,
            "keep_memmap": bool(keep_memmap) or pack.meta.get("storage") == "memmap",
            "project_dim": project_dim,
            "best_epoch": best_epoch,
            "best_l_hom": best_l_hom,
            "final_l_hom": float(hard["l_hom"]),
            "final_mean_sd_ortho": float(hard["mean_sd_ortho"]),
            "final_mean_sd_para": float(hard["mean_sd_para"]),
            "random_baseline_l_hom": baseline["l_hom"],
            "counts": counts,
            "n_nodes": pack.n_nodes,
            "n_features": int(pack.x.shape[1]),
            "min_epochs": min_epochs,
            "patience": patience,
            "max_epochs": max_epochs,
            "lambda_hom": lambda_hom,
            "lambda_size": lambda_size,
            "alpha_recon": alpha_recon,
            "beta_kl_max": beta_kl_max,
            "homology_in_encoder": False,
            "split_csv": str(split_csv),
            "model": "mlp_vae",
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
